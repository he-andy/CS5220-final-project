#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

struct discount_functor {
    double discount;
    discount_functor(double _discount) : discount(_discount) {}
    __device__ double operator()(const double& x) const {
        return x * discount;
    }
};

struct exercise_value_functor {
    double strike;
    exercise_value_functor(double _strike) : strike(_strike) {}
    __device__ double operator()(const double& x) const {
        return max(strike - x, 0.0);
    }
};

struct continuation_functor {
    double a, b, c;
    continuation_functor(double _a, double _b, double _c) : a(_a), b(_b), c(_c) {}
    __device__ double operator()(const double& x) const {
        return a * x * x + b * x + c;
    }
};

void quadratic_regression(double& a, double& b, double& c, thrust::device_vector<double>& x, thrust::device_vector<double>& y) {
    // make vandermonde matrix 
    int n = x.size();
    thrust::device_vector<double> A(n * 3);
    thrust::fill(A.begin(), A.begin() + n, 1.0);
    thrust::copy(x.begin(), x.end(), A.begin() + n);
    thrust::transform(A.begin() + n, A.begin() + 2 * n, A.begin() + n, A.begin() + 2 * n, thrust::multiplies<double>());

    thrust::device_vector<double> sol(3);


    double* d_A = thrust::raw_pointer_cast(A.data());
    double* d_y = thrust::raw_pointer_cast(y.data());
    double* d_x = thrust::raw_pointer_cast(sol.data());


    cusolverDnHandle_t cusolverH;
    cusolver_status = cusolverDnCreate(&cusolverH);

    // Query working space of geqrf and ormqr
    void* d_work;
    size_t work_size;

    cusolverDnDXgels_bufferSize(cusolverH, n, 3, 1, d_A, n, d_y, n, d_x, 3, dwork, &work_size);

    // Allocate working space
    cudaMalloc(&d_work, work_size);

    int* devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    int* niter;
    cudaMalloc(&niter, sizeof(int));

    // Solve the least squares problem
    cusolverDnDXgels(cusolverH, n, 3, 1, d_A, n, d_y, n, d_x, 3, d_work, work_size, niter, devInfo);

    int h_info = 0;
    cudaMemcpy(&h_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "QR decomposition failed!" << std::endl;
        return;
    }

    thrust::host_vector<double> h_y(sol.begin(), sol.end());
    a = h_y[0];
    b = h_y[1];
    c = h_y[2];

    cudaFree(dev_info);
    cudaFree(work);
    cusolverDnDestroy(cusolverH);
}

double ls_american_put_option_backward_pass(thrust::device_vector<thrust::device_vector<double>>& X, thrust::device_vector<int>& stop, double dt, double r, double strike) {
    int length = X.size();
    int paths = X[0].size();
    stop = thrust::device_vector<int>(paths, length - 1);
    double discount = exp(-r * dt);

    thrust::device_vector<double> cashflow = X[length - 1];
    thrust::transform(cashflow.begin(), cashflow.end(), cashflow.begin(), exercise_value_functor(strike));

    for (int i = length - 2; i > 0; i--) {
        // Discount the cashflows using thrust::transform
        thrust::transform(cashflow.begin(), cashflow.end(), cashflow.begin(), discount_functor(discount));

        thrust::device_vector<double> x = X[i];
        thrust::device_vector<double> exercise_value(paths);
        thrust::transform(x.begin(), x.end(), exercise_value.begin(), exercise_value_functor(strike));

        // find all the ITM paths
        thrust::device_vector<bool> itm(paths);
        thrust::transform(exercise_value.begin(), exercise_value.end(), itm.begin(), [] __device__(double ev) {
            return ev > 0;
        });

        int count = thrust::count(itm.begin(), itm.end(), true);

        if (count > 0) {
            // prune the paths that are not ITM
            thrust::device_vector<double> x_itm(count);
            thrust::device_vector<double> cashflow_itm(count);
            thrust::copy_if(x.begin(), x.end(), itm.begin(), x_itm.begin(), thrust::identity<bool>());
            thrust::copy_if(cashflow.begin(), cashflow.end(), itm.begin(), cashflow_itm.begin(), thrust::identity<bool>());

            //TODO: implement the quadratic regression for CUDA
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
                [] __device__(thrust::tuple<double, double, bool> t) {
                double exercise_value = thrust::get<0>(t);
                double continuation = thrust::get<1>(t);
                bool itm = thrust::get<2>(t);
                return itm && (exercise_value > continuation);
            }
            );

            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(cashflow.begin(), exercise_value.begin(), ex_idx.begin(), stop.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(cashflow.end(), exercise_value.end(), ex_idx.end(), stop.end())),
                thrust::make_zip_iterator(thrust::make_tuple(cashflow.begin(), stop.begin())),
                [i] __device__(thrust::tuple<double, double, bool, int> t) {
                double cashflow = thrust::get<0>(t);
                double exercise_value = thrust::get<1>(t);
                bool ex_idx = thrust::get<2>(t);
                int stop = thrust::get<3>(t);
                if (ex_idx) {
                    return thrust::make_tuple(exercise_value, i);
                }
                return thrust::make_tuple(cashflow, stop);
            }
            );
        }
    }
    // discount final time step
    thrust::transform(cashflow.begin(), cashflow.end(), cashflow.begin(), discount_functor(discount));

    return thrust::reduce(cashflow.begin(), cashflow.end()) / paths;
}