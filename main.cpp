#include "common.h"
#include <chrono>
#include <iostream>
#include <vector>

void simple_test() {
  // stock price paths as columns of X
  std::vector<std::vector<double>> X = {
      {1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00},
      {1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88},
      {1.08, 1.26, 1.07, 0.77, 1.56, 0.77, 0.84, 1.22},
      {1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34}};
  std::vector<double> t = {0, 1, 2, 3}; // time points
  double r = 0.06;                      // Risk-free rate
  double strike = 1.1;

  // Benchmark the function
  auto start_time = std::chrono::high_resolution_clock::now();
  ls_american_put_option_backward_pass(X, t, r, strike);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Execution time: " << duration.count() << " seconds"
            << std::endl;
}
int main() {
  simple_test();
  return 0;
}
