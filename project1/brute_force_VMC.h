#ifndef BRUTE_FORCE_VMC_H
#define BRUTE_FORCE_VMC_H

#include <cmath>

double wave_function(double r, double alpha);

double local_energy(double r, double alpha);

void monte_carlo_sampling(double *alphas,
                            int n_alphas,
                            double *energies,
                            double *variances,
                            int n_cycles);


#endif //BRUTE_FORCE_VMC_H
