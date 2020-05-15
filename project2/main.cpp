#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "sampler.h"
#include "Math/random.h"

using std::vector;
using std::cout;
using std::endl;


int main(){

    int nParticles = 1;
    int nDims = 1;
    int nHidden = 2;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    bool interaction = false;
    if(interaction) {assert(nParticles > 1);}


    int nMCcycles = 1e5; //number of montecarlo cycles
    int nOptimizeIters = 100; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double eta = 0.001; //learning rate
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed); //must be initialized first
    Hamiltonian hamiltonian(omega, interaction, nqs);
    Optimizer optimizer(eta, whichOptimizer);
    Sampler sampler(nMCcycles,
                    samplingRule,
                    tolerance,
                    nOptimizeIters,
                    stepLength,
                    timeStep,
                    hamiltonian,
                    nqs,
                    optimizer);
    //
    sampler.runOptimization();

    return 0;
}
