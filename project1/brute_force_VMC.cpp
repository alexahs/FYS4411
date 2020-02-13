#include "hamiltonian.h"
#include "random.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;


double wave_function(double r, double alpha){
    return exp(-0.5*alpha*alpha*r*r);
};


double local_energy(double r, double alpha){
    double alpha2 = alpha*alpha;
    return 0.5*(alpha2 + r*r*(1 - alpha2*alpha2));
};


void monte_carlo_sampling(double *alphas,
                            int n_alphas,
                            double *energies,
                            double *variances,
                            int n_cycles){

    double step_size = 1.0;
    double position_old = 0.0;
    double position_new = 0.0;
    // double *alpha_values = new double[max_variations];
    // double *energies = new double[n_cycles];


    for(int a = 0; a < n_alphas; a++){
    // for(double &alpha: alphas){
        // alpha_values[a] = alpha;
        // alpha += alpha_step;
        double energy = 0;
        double energy2 = 0;

        // cout << "alpha= " << alphas[a] << endl;


        position_old += step_size*(Random::nextDouble() - 0.5);
        double wf_old = wave_function(position_old, alphas[a]);


        for(int cycle = 0; cycle < n_cycles; cycle++){
            position_new = position_old + step_size*(Random::nextDouble() - 0.5);

            // cout << position_new << endl;

            double wf_new = wave_function(position_new, alphas[a]);
            if (Random::nextDouble() <= wf_new*wf_new / wf_old*wf_old){
                position_old = position_new;
                wf_old = wf_new;
            };
            double deltaE = local_energy(position_old, alphas[a]);
            energy += deltaE;
            energy2 += deltaE*deltaE;
        };
        energy /= n_cycles;
        energy2 /= n_cycles;
        double variance = energy2 - energy*energy;
        energies[a] = energy;
        variances[a] = variance;

    };
};
