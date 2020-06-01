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

void runGridSearch1p1d();
void runSingle();

int main(){


    // runSingle();
    runGridSearch1p1d();


    return 0;
}

void runSingle(){
    int nParticles = 1;
    int nDims = 1;
    int nHidden = 10;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 1.0; //initial spread of initial positions and weights
    bool interaction = false;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 12;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 100; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double eta = 0.1; //learning rate
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init); //must be initialized first
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
    sampler.runDataCollection(nMCcycles*8);
}


void runGridSearch1p1d(){

    int nParticles = 1;
    int nDims = 1;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.5; //initial spread of initial positions and weights
    bool interaction = false;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 14;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 100; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    std::vector<double> etaVals {0.5, 0.1, 0.01, 0.001, 0.0001};
    std::vector<int> hiddenVals {2, 4, 6, 8, 10};
    double eta;
    int nHidden;
    for(int i = 0; i < etaVals.size(); i++){
        for(int j = 0; j < hiddenVals.size(); j++){
            eta = etaVals[i];
            nHidden = hiddenVals[j];


            NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init); //must be initialized first
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
            sampler.m_printOptimInfo = false;
            sampler.runOptimization();
            sampler.runDataCollection(nMCcycles*8);
            sampler.printGridSearchInfo(i, j);

        }
    }

}
