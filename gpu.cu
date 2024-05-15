#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>

void quadratic_regression(double &a, double &b, double &c,
                          const std::vector<double> &x,
                          const std::vector<double> &y) {
  return;
}


__global__ void generate_path_kernel(double *paths, int n_paths, int n_time_steps,
                                    double initial_price, double delta_t,
                                    double drift, double volatility) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;

    curandState state;
    int seed = 0;
    curand_init(seed, idx, 0, &state);

    int offset = idx;
    paths[offset] = initial_price;
    for (int t = 1; t < n_time_steps; ++t) {
        offset += n_paths;
        double z = curand_normal_double(&state);
        double increment = sqrt(delta_t) * z;
        paths[offset] = paths[offset - n_paths] + drift * delta_t + volatility * increment;
    }
}

thrust::device_vector<double>
generate_random_paths(int n_paths, int n_time_steps, double initial_price,
                      double delta_t, double drift, double volatility) {

    thrust::device_vector<double> d_paths(n_paths * n_time_steps);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_paths + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    generate_path_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_paths.data()),
                                                            n_paths, n_time_steps, initial_price,
                                                            delta_t, drift, volatility);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return d_paths;
}

void test(int paths, int steps, double s0, double dt, double strike, double r,
          double drift, double vol, const std::string &save_path)
{
  std::vector<int> stop;
  auto start_time = std::chrono::high_resolution_clock::now();
  auto X = generate_random_paths(paths, steps, s0, dt, drift, vol);
  // // Benchmark the function
  // double price = ls_american_put_option_backward_pass(X, stop, dt, r, strike);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
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
    else
    {
      std::cerr << "Usage: " << argv[0]
                << " [-paths num] [-steps num] [-s0 value] [-dt value] "
                   "[-strike value] [-r rate] [-drift rate] [-vol volatility]"
                << std::endl;
      return 1;
    }
  }

  test(paths, steps, s0, dt, strike, r, drift, vol, save_path);

  return 0;
}
