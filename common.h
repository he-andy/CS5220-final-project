#include <iostream>
#include <vector>
#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Backward pass
double ls_american_put_option_backward_pass(std::vector<std::vector<double>> &X,
                                            std::vector<double> &t, double r,
                                            double strike);
// void simulate()
//
template <typename T> void printVector(const std::vector<T> &vec) {
  std::cout << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    std::cout << vec[i];
    if (i < vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
#endif
