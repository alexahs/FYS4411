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

double runSigmaGridSearch(int nCyclesPow2, int samplingRule, int whichOptimizer, double expected);
void runSingle();
void runGridSearch1(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma);
void runGridSearch2(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma);
void runGridSearch3(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma);
void runGridSearch4(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma);

int main(){

    ////////////////////////////////////////////////////////////////////////////

    double eta_min = 0.045;
    double eta_max = 0.2;
    int N_etas = 15;

    double hidden_min = 1;
    double hidden_max = 15;

    int nCyclesPow2 = 18;
    int samplingRule = 2;   //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme

    double expected = 3;    // Expected value in sigma search

    ////////////////////////////////////////////////////////////////////////////

    std::vector<double> etaVals;
    std::vector<int> hiddenVals;

    etaVals.assign(N_etas, 0);
    hiddenVals.assign(hidden_max - hidden_min + 1, 0);

    double d_eta = (eta_max - eta_min)/(N_etas-1);
    for (int i = 0; i < N_etas; i++) {
      etaVals[i] = eta_min + i*d_eta;
    }

    for (int i = 0; i < hidden_max - hidden_min + 1; i++) {
      hiddenVals[i] = hidden_min + i;
    }

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

    double sigma = runSigmaGridSearch(nCyclesPow2, samplingRule, whichOptimizer, expected);
    // Execution of main part of program
    auto t0 = std::chrono::high_resolution_clock::now();
    runGridSearch1(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals, sigma);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt1 = (t1 - t0);
    std::cout << "dt1 = " << dt1.count() << " s\n";
    runGridSearch2(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals, sigma);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt2 = (t2 - t1);
    std::cout << "dt2 = " << dt2.count() << " s\n";
    runGridSearch3(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals, sigma);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt3 = (t3 - t2);
    std::cout << "dt3 = " << dt3.count() << " s\n";
    runGridSearch4(nCyclesPow2, samplingRule, whichOptimizer, etaVals, hiddenVals, sigma);
    auto t4 = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> dt4 = (t4 - t3);
    std::cout << "dt4 = " << dt4.count() << " s\n";
    std::chrono::duration<double> dt5 = (t4 - t0);
    std::cout << "dt_total = " << dt5.count() << " s\n";

    return 0;
}

double runSigmaGridSearch(int nCyclesPow2, int samplingRule, int whichOptimizer, double expected) {
    int nParticles = 2;
    int nDims = 2;
    int nHidden = 2;

    double sigma_min = 1.12;
    double sigma_max = 1.16;
    int N_sigmas = 1000;

    double d_sigma = (sigma_max - sigma_min)/(N_sigmas-1);
    double sigmas [N_sigmas] = {};
    for (int i = 0; i < N_sigmas; i++) {
      sigmas[i] = sigma_min + i*d_sigma;
    }

    // SIZE IS HARDCODED!!!! CHECK THE LOOP TO BE SURE ALL IS CONSISTENT
    double meanEnergies [N_sigmas] = { };

    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.001; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double eta = 0.1; //learning rate
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 694206661337; //seed does nothing apparently

    int counter = 0;
    std::cout << "OPTIMIZING FOR SIGMA 0%" << std::flush;

    #pragma omp parallel for
    for (int i = 0; i < N_sigmas; i++) {
        NeuralQuantumState nqs(nParticles, nDims, nHidden, sigmas[i], seed, sigma_init, samplingRule); //must be initialized first
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
        sampler.runDataCollection(nMCcycles*8, false);
        // HARDCODED ABSOLUTE DIFFERENCE BETWEEN EXPECTED ENERGY AND MEAN
        meanEnergies[i] = sampler.getMeanEnergy();
        counter++;
        std::cout << "\rOPTIMIZING FOR SIGMA " << round(100*counter/N_sigmas) << "%" << std::flush;
    }
    double minimum = meanEnergies[0];
    int min_idx = 0;
    for (int i = 1; i < N_sigmas; i++) {
        if (abs(meanEnergies[i]-3) < minimum) {
            minimum = abs(meanEnergies[i]-expected);
            min_idx = i;
        }
    }
    std::cout << " – BEST SIGMA= " << sigmas[min_idx];
    std::cout << ", dE= " << minimum << std::endl;
    return sigmas[min_idx];
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

void runGridSearch1(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma){

    int nParticles = 2;
    int nDims = 2;
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

void runGridSearch2(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma){

    int nParticles = 2;
    int nDims = 1;
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

void runGridSearch3(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma){

    int nParticles = 2;
    int nDims = 2;
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

void runGridSearch4(int nCyclesPow2, int samplingRule, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma){

    int nParticles = 2;
    int nDims = 3;
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
