#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>

struct discount_functor
{
    double discount;
    discount_functor(double _discount) : discount(_discount) {}
    __device__ double operator()(const double &x) const
    {
        return x * discount;
    }
};

struct exercise_value_functor
{
    double strike;
    exercise_value_functor(double _strike) : strike(_strike) {}
    __device__ double operator()(const double &x) const
    {
        return max(strike - x, 0.0);
    }
};

struct continuation_functor
{
    double a, b, c;
    continuation_functor(double _a, double _b, double _c) : a(_a), b(_b), c(_c) {}
    __device__ double operator()(const double &x) const
    {
        return a * x * x + b * x + c;
    }
};

void quadratic_regression(double &a, double &b, double &c, thrust::device_vector<double> &x, thrust::device_vector<double> &y)
{
    // make vandermonde matrix
    int n = x.size();
    thrust::device_vector<double> A(n * 3);
    thrust::fill(A.begin(), A.begin() + n, 1.0);
    thrust::copy(x.begin(), x.end(), A.begin() + n);
    thrust::transform(A.begin() + n, A.begin() + 2 * n, A.begin() + n, A.begin() + 2 * n, thrust::multiplies<double>());

    thrust::device_vector<double> sol(3);

    double *d_A = thrust::raw_pointer_cast(A.data());
    double *d_y = thrust::raw_pointer_cast(y.data());
    double *d_x = thrust::raw_pointer_cast(sol.data());

    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // Query working space of geqrf and ormqr
    void *d_work = NULL;
    size_t work_size = 0;

    cusolverDnDDgels_bufferSize(cusolverH, n, 3, 1, d_A, n, d_y, n, d_x, 3, d_work, &work_size);
    // Allocate working space
    cudaMalloc(&d_work, work_size);

    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    int niter;
    // cudaMalloc(&niter, sizeof(int));

    // Solve the least squares problem
    cusolverDnDDgels(cusolverH, n, 3, 1, d_A, n, d_y, n, d_x, 3, d_work, work_size, &niter, devInfo);

    int h_info = 0;
    cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0)
    {
        std::cerr << "QR decomposition failed!" << std::endl;
        return;
    }

    thrust::host_vector<double> h_y(sol.begin(), sol.end());
    a = h_y[0];
    b = h_y[1];
    c = h_y[2];

    cudaFree(devInfo);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}

double ls_american_put_option_backward_pass(thrust::device_vector<double> &X, thrust::device_vector<int> &stop, int length, int paths, double dt, double r, double strike)
{
    double discount = exp(-r * dt);

    thrust::device_vector<double> cashflow(paths);
    thrust::transform(X.begin() + (length - 1) * paths, X.begin() + length * paths, cashflow.begin(), exercise_value_functor(strike));

    for (int i = length - 2; i > 0; i--)
    {
        // Discount the cashflows using thrust::transform
        thrust::transform(cashflow.begin(), cashflow.end(), cashflow.begin(), discount_functor(discount));

        thrust::device_vector<double> x(paths);
        thrust::copy(X.begin() + i * paths, X.begin() + (i + 1) * paths, x.begin());
        thrust::device_vector<double> exercise_value(paths);
        thrust::transform(x.begin(), x.end(), exercise_value.begin(), exercise_value_functor(strike));

        // find all the ITM paths
        thrust::device_vector<bool> itm(paths);
        thrust::transform(exercise_value.begin(), exercise_value.end(), itm.begin(), [] __device__(double ev)
                          { return ev > 0; });

        int count = thrust::count(itm.begin(), itm.end(), true);

        if (count > 0)
        {
            // prune the paths that are not ITM
            thrust::device_vector<double> x_itm(count);
            thrust::device_vector<double> cashflow_itm(count);
            thrust::copy_if(x.begin(), x.end(), itm.begin(), x_itm.begin(), thrust::identity<bool>());
            thrust::copy_if(cashflow.begin(), cashflow.end(), itm.begin(), cashflow_itm.begin(), thrust::identity<bool>());

            // TODO: implement the quadratic regression for CUDA
            double a, b, c;
            quadratic_regression(a, b, c, x_itm, cashflow_itm);

            // compute continuation values
            thrust::device_vector<double> continuation(paths);
            thrust::transform(x.begin(), x.end(), continuation.begin(), continuation_functor(a, b, c));

            // find the paths that should be exercised
            // equivalent to ex_idx[j] = itm[j] && (exercise_value[j] > continuation[j])
            thrust::device_vector<bool> ex_idx(paths);
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(exercise_value.begin(), continuation.begin(), itm.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(exercise_value.end(), continuation.end(), itm.end())),
                ex_idx.begin(),
                [] __device__(thrust::tuple<double, double, bool> t)
                {
                    double exercise_value = thrust::get<0>(t);
                    double continuation = thrust::get<1>(t);
                    bool itm = thrust::get<2>(t);
                    return itm && (exercise_value > continuation);
                });

            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(cashflow.begin(), exercise_value.begin(), ex_idx.begin(), stop.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(cashflow.end(), exercise_value.end(), ex_idx.end(), stop.end())),
                thrust::make_zip_iterator(thrust::make_tuple(cashflow.begin(), stop.begin())),
                [i] __device__(thrust::tuple<double, double, bool, int> t)
                {
                    double cashflow = thrust::get<0>(t);
                    double exercise_value = thrust::get<1>(t);
                    bool ex_idx = thrust::get<2>(t);
                    int stop = thrust::get<3>(t);
                    if (ex_idx)
                    {
                        return thrust::make_tuple(exercise_value, i);
                    }
                    return thrust::make_tuple(cashflow, stop);
                });
        }
    }
    // discount final time step
    thrust::transform(cashflow.begin(), cashflow.end(), cashflow.begin(), discount_functor(discount));

    return thrust::reduce(cashflow.begin(), cashflow.end()) / paths;
}

__global__ void generate_path_kernel(double *paths, int n_paths, int n_time_steps,
                                     double initial_price, double delta_t,
                                     double drift, double volatility, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths)
        return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int offset = idx;
    paths[offset] = initial_price;
    for (int t = 1; t < n_time_steps; ++t)
    {
        offset += n_paths;
        double z = curand_normal_double(&state);
        double increment = sqrt(delta_t) * z;
        paths[offset] = paths[offset - n_paths] + drift * delta_t + volatility * increment;
    }
}

thrust::device_vector<double>
generate_random_paths(int n_paths, int n_time_steps, double initial_price,
                      double delta_t, double drift, double volatility, int seed)
{

    thrust::device_vector<double> d_paths(n_paths * n_time_steps);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_paths + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    generate_path_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_paths.data()),
                                                             n_paths, n_time_steps, initial_price,
                                                             delta_t, drift, volatility, seed);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return d_paths;
}

// thrust::device_vector<double>
// generate_random_paths(int n_paths, int n_time_steps, double initial_price,
//                       double delta_t, double drift, double volatility, int seed)
// {
//     std::vector<double> matrix(n_time_steps * n_paths, initial_price);
//     std::vector<std::mt19937> generators;
//     std::vector<std::normal_distribution<double>> distributions;

//     generators.resize(n_paths);
//     distributions.resize(n_paths);

//     std::mt19937 seed_gen(seed);
//     std::uniform_int_distribution<int> seed_distribution(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());

//     // serial seeding of generators
//     for (int p = 0; p < n_paths; p++)
//     {
//         int path_seed = seed_distribution(seed_gen);
//         std::mt19937 tmp(path_seed);
//         generators[p] = tmp;
//         // std::cout << "path " << p << " seeded with " << path_seed << std::endl;
//     }
// #pragma omp parallel for
//     for (int p = 0; p < n_paths; p++)
//     {
//         distributions[p] = std::normal_distribution<double>(0.0, 1.0);
//     }

//     for (int t = 1; t < n_time_steps; ++t)
//     {
// #pragma omp parallel for
//         for (int p = 0; p < n_paths; p++)
//         {
//             const double sample = distributions[p](generators[p]);
//             const double increment = std::sqrt(delta_t) * sample;
//             matrix[t * n_paths + p] =
//                 matrix[(t - 1) * n_paths + p] + drift * delta_t + volatility * increment;
//         }
//     }

//     thrust::device_vector<double> d_matrix(matrix.begin(), matrix.end());
//     return d_matrix;
// }

void test(int paths, int steps, double s0, double dt, double strike, double r,
          double drift, double vol, const std::string &save_path, int seed)
{
    thrust::device_vector<int> stop(paths, 0);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto X = generate_random_paths(paths, steps, s0, dt, drift, vol, seed);
    std::cout << "Generated paths" << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    // std::cout << "Price: " << price << std::endl;
    std::cout << "Execution time paths: " << duration.count() << " seconds"
              << std::endl;

    auto start_time2 = std::chrono::high_resolution_clock::now();
    // // Benchmark the function
    double price = ls_american_put_option_backward_pass(X, stop, steps, paths, dt, r, strike);
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time2;
    // std::cout << "Price: " << price << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds"
              << std::endl;

    // if (save_path != "")
    // {
    //   std::vector<double> to_save(paths * steps / save_freq);
    //   for (int i = 0; i < steps; i += save_freq)
    //   {
    //     std::copy(X[i].begin(), X[i].end(), to_save.begin() + i / save_freq * paths);
    //   }

    //   std::ofstream outfile(save_path, std::ios::out | std::ios::binary);

    //   if (!outfile)
    //   {
    //     std::cout << "Could not save steps..." << std::endl;
    //     return;
    //   }

    //   outfile.write(reinterpret_cast<const char *>(to_save.data()), to_save.size() * sizeof(double));
    //   outfile.close();
    // }
}

int main(int argc, char *argv[])
{
    // Default values for the parameters
    int paths = 10000;
    int steps = 100;
    double s0 = 100.0;
    double dt = 0.01;
    double strike = 100.0;
    double r = 0.05;
    double drift = 0.05;
    double vol = 0.2;
    int seed = std::random_device{}();
    std::string save_path = "";

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-paths" && i + 1 < argc)
            paths = std::atoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc)
            steps = std::atoi(argv[++i]);
        else if (arg == "-s0" && i + 1 < argc)
            s0 = std::atof(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc)
            dt = std::atof(argv[++i]);
        else if (arg == "-strike" && i + 1 < argc)
            strike = std::atof(argv[++i]);
        else if (arg == "-r" && i + 1 < argc)
            r = std::atof(argv[++i]);
        else if (arg == "-drift" && i + 1 < argc)
            drift = std::atof(argv[++i]);
        else if (arg == "-vol" && i + 1 < argc)
            vol = std::atof(argv[++i]);
        else if (arg == "-save" && i + 1 < argc)
            save_path = argv[++i];
        else if (arg == "-seed" && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else
        {
            std::cerr << "Usage: " << argv[0]
                      << " [-paths num] [-steps num] [-s0 value] [-dt value] "
                         "[-strike value] [-r rate] [-drift rate] [-vol volatility]"
                      << std::endl;
            return 1;
        }
    }

    test(paths, steps, s0, dt, strike, r, drift, vol, save_path, seed);

    return 0;
}
