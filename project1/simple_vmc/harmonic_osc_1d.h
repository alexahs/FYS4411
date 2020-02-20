#ifndef HARMONIC_OSC_1D_H
#define HARMONIC_OSC_1D_H

#include <cmath>

void run(int n_cycles);

double wave_function(double r, double alpha);

double local_energy(double r, double alpha);

void monte_carlo_sampling(double *alphas,
                            int n_alphas,
                            double *energies,
                            double *variances,
                            int n_cycles);


#endif //HARMONIC_OSC_1D_H
