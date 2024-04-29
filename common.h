#include <vector>
#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Backward pass
double ls_american_put_option_backward_pass(std::vector<std::vector<double>> &X,
                                            std::vector<double> &t, double r,
                                            double strike);
// void simulate()
#endif
