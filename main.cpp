#include "common.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

void simple_test()
{
  // stock price paths as columns of X
  std::vector<std::vector<double>> X = {
      {1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00},
      {1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88},
      {1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.83, 1.22},
      {1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34}};
  double r = 0.06; // Risk-free rate
  double strike = 1.1;
  std::vector<int> stop;

  // Benchmark the function
  auto start_time = std::chrono::high_resolution_clock::now();
  double price = ls_american_put_option_backward_pass(X, stop, 1.0, r, strike);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "Price: " << price << std::endl;
  std::cout << "Execution time: " << duration.count() << " seconds"
            << std::endl;
}

void test(int paths, int steps, double s0, double dt, double strike, double r,
          double drift, double vol, const std::string &save_path, int seed)
{
  std::vector<int> stop;
  auto start_time = std::chrono::high_resolution_clock::now();
  auto X = generate_random_paths(paths, steps, s0, dt, drift, vol, seed);
  // Benchmark the function
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "Generate paths time: " << duration.count() << " seconds" << std::endl;
  double price = ls_american_put_option_backward_pass(X, stop, dt, r, strike);
  end_time = std::chrono::high_resolution_clock::now();
  duration = end_time - start_time;
  std::cout << "Price: " << price << std::endl;
  std::cout << "Execution time: " << duration.count() << " seconds"
            << std::endl;

  if (save_path != "")
  {
    std::vector<double> to_save(paths * steps / save_freq);
    for (int i = 0; i < steps; i += save_freq)
    {
      std::copy(X[i].begin(), X[i].end(), to_save.begin() + i / save_freq * paths);
    }

    std::ofstream outfile(save_path, std::ios::out | std::ios::binary);

    if (!outfile)
    {
      std::cout << "Could not save steps..." << std::endl;
      return;
    }

    outfile.write(reinterpret_cast<const char *>(to_save.data()), to_save.size() * sizeof(double));
    outfile.close();
  }
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
