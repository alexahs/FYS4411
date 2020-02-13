#include <iostream>
#include "Hamiltonians/hamiltonian.h"
#include "random.h"
#include "brute_force_VMC.h"
#include <fstream>
#include <iomanip>

using namespace std;

int main() {

    int n_alphas = 20;
    int n_cycles = 100000;
    double *alphas = new double[n_alphas];
    double *energies = new double[n_alphas];
    double *variances = new double[n_alphas];

    double alpha = 0.4;

    for(int i = 0; i < n_alphas; i++){
        alphas[i] = alpha;
        alpha += 0.05;
    }

    monte_carlo_sampling(alphas, n_alphas, energies, variances, n_cycles);


    cout << setw(15) << "alpha";
    cout << setw(15) << "energy";
    cout << setw(15) << "variance" << endl;

    for(int i = 0; i < n_alphas; i++){
        cout << setw(15) << setprecision(8) << alphas[i];
        cout << setw(15) << setprecision(8) << energies[i];
        cout << setw(15) << setprecision(8) << variances[i] << endl;
    }




    return 0;
}


// double monte_carlo_sampling(double *alphas,
//                             int n_alphas,
//                             double *energies,
//                             double *variances,
//                             int n_cycles);
