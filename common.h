#include <vector>
#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Backward pass
double ls_american_put_option_backward_pass(std::vector<std::vector<double>> &X,
                                            std::vector<double> &t, double r,
                                            double strike);
// void simulate()
std::vector<std::vector<double>> generate_random_paths(int n_paths, int n_time_steps, double initial_price, double delta_t, double drift, double volatility);
#endif
