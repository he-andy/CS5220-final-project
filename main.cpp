#include "common.h"
#include <chrono>
#include <iostream>
#include <vector>

void simple_test()
{
  // stock price paths as columns of X
  // std::vector<std::vector<double>> X = {
  //     {1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00},
  //     {1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88},
  //     {1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.83, 1.22},
  //     {1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34}};
  auto X = generate_random_paths(8, 4, 1.0, 1.0, 1.0, 1.0);

  // std::vector<std::vector<double>> X = {
  //     {1, 1, 1, 1, 1, 1, 1, 1},
  //     {2.4097, 1.75248, 2.91011, 1.39089, 1.36223, 2.76846, 0.611664, 1.999},
  //     {0.584831, 3.30778, 3.88314, 1.4821, 3.43574, 4.26843, 1.56471, 3.59584},
  //     {1.84363, 5.1218, 4.61117, 2.9252, 2.52425, 6.80687, 2.9651, 5.52296}};

  std::cout << X.size() << ", " << X[0].size() << std::endl;
  for (auto &row : X)
  {
    for (auto &entry : row)
    {
      std::cout << entry << ", ";
    }
    std::cout << std::endl;
  }
  std::vector<double> t = {0, 1, 2, 3}; // time points
  double r = 0.06;                      // Risk-free rate
  double strike = 1.1;

  // Benchmark the function
  auto start_time = std::chrono::high_resolution_clock::now();
  double price = ls_american_put_option_backward_pass(X, t, r, strike);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "Price: " << price << std::endl;
  std::cout << "Execution time: " << duration.count() << " seconds"
            << std::endl;
}

int main()
{
  simple_test();
  return 0;
}
