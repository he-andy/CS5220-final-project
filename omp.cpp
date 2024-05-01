#include "common.h"
#include <cblas.h>
// https://docs.nersc.gov/development/libraries/lapack/
#include <lapacke.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

void quadratic_regression(double &a, double &b, double &c,
                          const std::vector<double> &x,
                          const std::vector<double> &y)
{
  int n = x.size();
  std::vector<double> A(n * 3);
  for (int i = 0; i < n; ++i)
  {
    A[i] = 1.0;
    A[i + n] = x[i];
    A[i + 2 * n] = x[i] * x[i];
  }

  int lda = n;
  int m = 3;
  int nrhs = 1;
  int ldb = std::max(3, n);
  int info;
  int rank;
  double minus_one = -1.0;
  std::vector<double> S(3); // At most 3 singular values
  std::vector<double> Y(ldb);
  // Copy the y values
  std::copy(y.begin(), y.end(), Y.begin());
  // Workspace query
  double wkopt;
  int iwkopt;
  int lwork = -1;
  int liwork = -1;
  // Query optimal workspace
  dgelsd_(&n, &m, &nrhs, A.data(), &lda, Y.data(), &ldb, S.data(), &minus_one,
          &rank, &wkopt, &lwork, &iwkopt, &info);

  lwork = (int)wkopt;
  liwork = iwkopt;
  std::vector<double> work(lwork);
  std::vector<int> iwork(liwork);
  // Compute the solution
  dgelsd_(&n, &m, &nrhs, A.data(), &lda, Y.data(), &ldb, S.data(), &minus_one,
          &rank, work.data(), &lwork, iwork.data(), &info);
  if (info != 0)
  {
    std::cerr << "The algorithm computing SVD failed to converge, info: "
              << info << std::endl;
    return;
  }

  // Retrieve the coefficients
  c = Y[0];
  b = Y[1];
  a = Y[2];
}

double eval_quadratic(double a, double b, double c, double x)
{
  // return a * x * x + b * x + c;
  return c + x * (b + a * x);
}

double ls_american_put_option_backward_pass(std::vector<std::vector<double>> &X, std::vector<int> &stop,
                                            double dt, double r,
                                            double strike)
{
  int length = X.size();
  int paths = X[0].size();
  stop = std::vector<int>(paths, length - 1);
  double discount = exp(-r * dt);

  std::vector<double> cashflow = std::move(X[length - 1]);
  for (int i = 0; i < paths; i++)
  {
    cashflow[i] = std::max(strike - cashflow[i], 0.0);
  }

  for (int i = length - 2; i > 0; i--)
  {
    // compute discount factor
    // discount cashflow for this timestep
    cblas_dscal(paths, discount, cashflow.data(), 1);
    std::vector<double> x = std::move(X[i]);
    // exercise values for this timestep
    std::vector<double> exercise_value(paths);
    for (int j = 0; j < paths; j++)
    {
      exercise_value[j] = std::max(strike - x[j], 0.0);
    }

    std::vector<bool> itm(paths);
    int count = 0;
    for (int j = 0; j < paths; j++)
    {
      itm[j] = exercise_value[j] > 0;
      if (itm[j])
      {
        count++;
      }
    }
    // prune the paths that are not in the money
    // note, i think there are very fast CUDA kernel implementations for this
    std::vector<double> x_itm(count);
    std::vector<double> cashflow_itm(count);
    int k = 0;
    for (int j = 0; j < paths; j++)
    {
      if (itm[j])
      {
        x_itm[k] = x[j];
        cashflow_itm[k] = cashflow[j];
        k += 1;
      }
    }
    std::vector<double> continuation(paths);
    std::vector<bool> ex_idx(paths);
    // if there are ITM paths
    if (k != 0)
    {
      double a, b, c;
      quadratic_regression(a, b, c, x_itm, cashflow_itm);

      for (int j = 0; j < paths; j++)
      {
        continuation[j] = eval_quadratic(c, b, a, x[j]);
      }
      for (int j = 0; j < paths; j++)
      {
        ex_idx[j] = itm[j] && (exercise_value[j] > continuation[j]);
      }
    }
    // there are no ITM paths, so we don't exercise
    else
    {
      std::fill(ex_idx.begin(), ex_idx.end(), false);
    }

    for (int j = 0; j < paths; j++)
    {
      if (ex_idx[j])
      {
        cashflow[j] = exercise_value[j];
        stop[j] = i;
      }
    }
  }

  // discount the final timestep
  cblas_dscal(paths, discount, cashflow.data(), 1);
  // return mean of cashflows at t0
  double sum = 0.0;
  for (int i = 0; i < paths; i++)
  {
    sum += cashflow[i];
  }
  return sum / paths;
}

std::vector<std::vector<double>>
generate_random_paths(int n_paths, int n_time_steps, double initial_price,
                      double delta_t, double drift, double volatility)
{
  std::vector<std::vector<double>> matrix(
      n_time_steps, std::vector<double>(n_paths, initial_price));
  std::vector<std::mt19937> generators;
  std::vector<std::normal_distribution<double>> distributions;
  int n_threads = omp_get_num_threads();

  generators.resize(n_threads);
  distributions.resize(n_threads);
#pragma omp parallel for
  for (int tid = 0; tid < n_threads; tid++)
  {
    generators[tid] = std::mt19937{std::random_device{}()};
    distributions[tid] = std::normal_distribution<double>(0.0, 1.0);
  }

  for (int t = 1; t < n_time_steps; ++t)
  {
#pragma omp parallel for
    for (int p = 0; p < n_paths; p++)
    {
      const int tid = omp_get_thread_num();
      const double sample = distributions[tid](generators[tid]);
      const double increment = std::sqrt(delta_t) * sample;
      matrix[t][p] =
          matrix[t - 1][p] + drift * delta_t + volatility * increment;
    }
  }

  return matrix;
}
