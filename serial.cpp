#include "common.h"
#include <cblas.h>
// https://docs.nersc.gov/development/libraries/lapack/
#include <lapacke.h>

#include <cmath>

#include "lapacke.h" // Make sure LAPACK is properly linked
#include <iostream>
#include <vector>

// void quadratic_regression(double &a, double &b, double &c,
//                           const std::vector<double> &x,
//                           const std::vector<double> &y) {
//   // Number of data points
//   int n = x.size();

//   // Number of coefficients in the polynomial
//   int m = 3;

//   // Set up the matrix A with dimensions nxm
//   double A[n * m];
//   for (int i = 0; i < n; ++i) {
//     A[i + n * 0] = x[i] * x[i]; // x^2
//     A[i + n * 1] = x[i];        // x
//     A[i + n * 2] = 1;           // 1
//   }

//   // Set up the vector b
//   double B[n];
//   for (int i = 0; i < n; ++i) {
//     B[i] = y[i];
//   }

//   // Solve the least squares problem using LAPACK
//   int lda = n;
//   int ldb = n;
//   int nrhs = 1;
//   int info;
//   int lwork = -1;
//   double wkopt;
//   double minus_one = -1.0;
//   // Query and allocate the optimal workspace
//   dgelss_(&n, &m, &nrhs, A, &lda, B, &ldb, nullptr, &minus_one, &wkopt,
//   &lwork,
//           &info);
//   lwork = (int)wkopt;
//   double work[lwork];

//   // Solve the equation A*X = B
//   double singular_values[m];
//   dgelss_(&n, &m, &nrhs, A, &lda, B, &ldb, singular_values, &minus_one, work,
//           &lwork, &info);

//   // Check for successful completion
//   if (info > 0) {
//     std::cerr << "The algorithm computing SVD failed to converge." <<
//     std::endl; exit(1);
//   }

//   // Output the coefficients
//   a = B[0];
//   b = B[1];
//   c = B[2];
// }

void quadratic_regression(double &a, double &b, double &c,
                          const std::vector<double> &x,
                          const std::vector<double> &y) {
  int n = x.size();

  std::vector<double> A(n * 3);
  for (int i = 0; i < n; ++i) {
    A[i] = 1.0;
    A[i + n] = x[i];
    A[i + 2 * n] = x[i] * x[i];
  }

  int lda = n;
  int m = 3;
  int nrhs = 1;
  int ldb = n;
  int info;
  int rank;
  double minus_one = -1.0;
  std::vector<double> S(3); // At most 3 singular values
  std::vector<double> Y(y);
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
  if (info != 0) {
    std::cerr << "The algorithm computing SVD failed to converge, info: "
              << info << std::endl;
    return;
  }

  // Retrieve the coefficients
  c = Y[0];
  b = Y[1];
  a = Y[2];
}

double eval_quadratic(double a, double b, double c, double x) {
  return a * x * x + b * x + c;
}

double ls_american_put_option_backward_pass(std::vector<std::vector<double>> &X,
                                            std::vector<double> &t, double r,
                                            double strike) {
  int length = X.size();
  int paths = X[0].size();

  std::vector<double> cashflow = std::move(X[length - 1]);
  for (int i = 0; i < paths; i++) {
    cashflow[i] = std::max(strike - cashflow[i], 0.0);
  }

  for (int i = length - 2; i > 0; i--) {
    // compute discount factor
    double dt = t[i + 1] - t[i];
    double discount = exp(-r * dt);
    // discount cashflow for this timestep
    cblas_dscal(paths, discount, cashflow.data(), 1);
    std::vector<double> x = std::move(X[i]);
    // exercise values for this timestep
    std::vector<double> exercise_value(paths);
    for (int j = 0; j < paths; j++) {
      exercise_value[j] = std::max(strike - x[j], 0.0);
    }

    std::vector<bool> itm(paths);
    int count = 0;
    for (int j = 0; j < paths; j++) {
      itm[j] = exercise_value[j] > 0;
      if (itm[j]) {
        count++;
      }
    }
    // prune the paths that are not in the money
    // note, i think there are very fast CUDA kernel implementations for this
    std::vector<double> x_itm(count);
    std::vector<double> cashflow_itm(count);
    int k = 0;
    for (int j = 0; j < paths; j++) {
      if (itm[j]) {
        x_itm[k] = x[j];
        cashflow_itm[k] = cashflow[j];
        k += 1;
      }
    }

    double a, b, c;
    quadratic_regression(a, b, c, x_itm, cashflow_itm);
    std::vector<double> continuation(paths);

    for (int j = 0; j < paths; j++) {
      continuation[j] = eval_quadratic(c, b, a, x[j]);
    }

    std::vector<bool> ex_idx(paths);
    for (int j = 0; j < paths; j++) {
      ex_idx[j] = itm[j] && (exercise_value[j] > continuation[j]);
    }

    for (int j = 0; j < paths; j++) {
      if (ex_idx[j]) {
        cashflow[j] = exercise_value[j];
      }
    }
  }

  // discount the final timestep
  double dt = t[1] - t[0];
  double discount = exp(-r * dt);
  cblas_dscal(paths, discount, cashflow.data(), 1);
  // return mean of cashflows at t0
  double sum = 0.0;
  for (int i = 0; i < paths; i++) {
    sum += cashflow[i];
  }
  return sum / paths;
}
