#include <iostream>
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
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double eta = 0.001; //learning rate
    double tolerance = 1e-6; 


    Hamiltonian hamiltonian(omega);
    NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma);
    Optimizer optimizer(eta, whichOptimizer);
    Sampler sampler(nMCcycles,
                      nOptimizeIters,
                      samplingRule,
                      tolerance,
                      hamiltonian,
                      nqs,
                      optimizer);
    //
    sampler.runOptimization();

    return 1;
}
