#include <iosteam>
#include <vector>
#include <iomanip>

#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "sampler.h"
#include "Math/random.h"


int main(){

    int nParticles = 2;
    int nDims = 2;
    int nHidden = 10;
    double sigma = 1;
    double omega = 1;

    int nMCcycles = 1000;
    int nOptimizeIters = 100;
    int samplingRule = 1; //1 - standard, 2 - metropolis, 3- gibbs


    hamiltonian = Hamiltonian(omega);
    nqs = NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma);
    optimizer = Optimizer(); //implement
    sampler = Sampler(nMCcycles,
                      nOptimizeIters,
                      samplingRule,
                      &hamiltonian,
                      &nqs,
                      &optimizer);
    //
    sampler.runOptimization();

    return 1;
}
