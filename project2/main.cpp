#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <chrono>
#include <omp.h>

#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "sampler.h"
#include "Math/random.h"

using std::vector;
using std::cout;
using std::endl;

void runSingle();
void runGridSearch1(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals);
void runGridSearch2(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals);
void runGridSearch3(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals);
void runGridSearch4(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals);

int main(){

    // std::vector<double> etaVals {0.09, 0.095, 0.10, 0.105, 0.11};
    std::vector<double> etaVals {0.045, 0.056, 0.067, 0.078, 0.089, 0.100, 0.111, 0.123, 0.134, 0.145, 0.156, 0.167, 0.178, 0.189, 0.200};
    // std::vector<int> hiddenVals {1,2,3,4,5};
    std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int nCyclesPow2 = 15;
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme

    while (true) {
        std::cout << "Delete Previous Data? (y/n/q)" << std::endl;
        char input;
        std::cin >> input;
        if (input == 'y' || input == 'Y') {
          int output = system("rm -rf Data/*");
          break;
        } else if (input == 'q' || input == 'Q') {
          exit(0);
        } else if (input != 'n' && input != 'N') {
          std::cout << "Invalid Input: " << input << std::endl;
        } else {
          break;
        }
    }

    runSingle();

    // runSingle();
    // Execution of main part of program
    // auto t0 = std::chrono::high_resolution_clock::now();
    // runGridSearch1(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> dt1 = (t1 - t0);
    // std::cout << "dt1 = " << dt1.count() << " s\n";
    // runGridSearch2(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> dt2 = (t2 - t1);
    // std::cout << "dt2 = " << dt2.count() << " s\n";
    // runGridSearch3(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals);
    // auto t3 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> dt3 = (t3 - t2);
    // std::cout << "dt3 = " << dt3.count() << " s\n";
    // runGridSearch4(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals);
    // auto t4 = std::chrono::high_resolution_clock::now();
  	// std::chrono::duration<double> dt4 = (t4 - t3);
    // std::cout << "dt4 = " << dt4.count() << " s\n";
    // std::chrono::duration<double> dt5 = (t4 - t0);
    // std::cout << "dt_total = " << dt5.count() << " s\n";

    return 0;
}

void runSingle(){
    int nParticles = 2;
    int nDims = 2;
    int nHidden = 2;
    double sigma = 1.5; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.001; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 15;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 3; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double eta = 0.1; //learning rate
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 694206661337; //seed does nothing apparently

    NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init, samplingRule); //must be initialized first
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

void runGridSearch1(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals){

    int nParticles = 2;
    int nDims = 2;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.001; //initial spread of initial positions and weights
    bool interaction = false;
    if(interaction) {assert(nParticles > 1);}

    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 100; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently

    double eta;
    int nHidden;
    double counter = 0.0;
    double tot_iter = etaVals.size()*hiddenVals.size();
    std::cout << "LOADING 0%" << std::flush;
    for(int i = 0; i < etaVals.size(); i++){
        #pragma omp parallel for
        for(int j = 0; j < hiddenVals.size(); j++){
            eta = etaVals[i];
            nHidden = hiddenVals[j];
            NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init, samplingRule); //must be initialized first
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;
}

void runGridSearch2(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals){

    int nParticles = 2;
    int nDims = 1;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.001; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently

    double eta;
    int nHidden;
    double counter = 0.0;
    double tot_iter = etaVals.size()*hiddenVals.size();
    std::cout << "LOADING 0%" << std::flush;
    for(int i = 0; i < etaVals.size(); i++){
        #pragma omp parallel for
        for(int j = 0; j < hiddenVals.size(); j++){
            eta = etaVals[i];
            nHidden = hiddenVals[j];

            NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init, samplingRule); //must be initialized first
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;
}

void runGridSearch3(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals){

    int nParticles = 2;
    int nDims = 2;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.001; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently, seed is set in Math/random.cpp


    double eta;
    int nHidden;
    double counter = 0.0;
    double tot_iter = etaVals.size()*hiddenVals.size();
    std::cout << "LOADING 0%" << std::flush;
    for(int i = 0; i < etaVals.size(); i++){
        #pragma omp parallel for
        for(int j = 0; j < hiddenVals.size(); j++){
            eta = etaVals[i];
            nHidden = hiddenVals[j];

            NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init, samplingRule); //must be initialized first
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;

}

void runGridSearch4(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals){

    int nParticles = 2;
    int nDims = 3;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.001; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently

    double eta;
    int nHidden;
    double counter = 0.0;
    double tot_iter = etaVals.size()*hiddenVals.size();
    std::cout << "LOADING 0%" << std::flush;
    for(int i = 0; i < etaVals.size(); i++){
        #pragma omp parallel for
        for(int j = 0; j < hiddenVals.size(); j++){
            eta = etaVals[i];
            nHidden = hiddenVals[j];

            NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigma_init, samplingRule); //must be initialized first
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
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;

}
