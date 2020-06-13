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

double runSigmaSearch(double sigma_min, double sigma_max, int nCyclesPow2, int samplingRule, int whichOptimizer, double expected, int nParticles, int nDims, bool interaction);
double runSigmaInitSearch(double sigma_min, double sigma_max, int nCyclesPow2, int samplingRule, int whichOptimizer, double expected, int nParticles, int nDims, double sigma, bool interaction);
void runSingle();
void runGridSearch(int nParticles, int nDims, bool interaction, int samplingRule,
                   int nCyclesPow2, int whichOptimizer, std::vector<double> etaVals,
                   std::vector<int> hiddenVals, double sigma, double sigma_init,
                   int nOptimizeIters, double stepLength, double timeStep,
                   double tolerance, double omega);

int main(){

  ////////////////////////
  // CONFIRM DATA RESET //
  ////////////////////////

    system("mkdir -p Data");
    while (true) {
        std::cout << "Delete Previous Data? (y/n/q)" << std::endl;
        char input;
        std::cin >> input;
        if (input == 'y' || input == 'Y') {
          system("rm -rf Data/*");
          break;
        } else if (input == 'q' || input == 'Q') {
          exit(0);
        } else if (input != 'n' && input != 'N') {
          std::cout << "Invalid Input: " << input << std::endl;
        } else {
          break;
        }
    }

  ///////////////////
  // GLOBAL PARAMS //
  ///////////////////

    double omega = 1.0; //in hamiltonian
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double tolerance = 1e-6; //tolerance for convergence
    int whichOptimizer = 1;       //1 - gradient descent, 2 - some other optim scheme

  ///////////////////////
  // GRIDSEARCH PARAMS //
  ///////////////////////

    double eta_min = 0.045;
    double eta_max = 0.2;
    int N_etas = 15;

    double hidden_min = 1;
    double hidden_max = 15;

    int nCyclesPow2 = 13;
    int nCyclesSigma = 8;
    // int nCyclesPow2 = 6;          // For testing
    // int nCyclesSigma = 4;         // For testing

    double expected_1P = 0.5;     // Expected value in sigma search
    double expected_2P = 3.0;     // Expected value in sigma search

  //////////////////////////////////////////////
  // DETERMINING IDEAL SIGMAS AND SIGMA INITS //
  //////////////////////////////////////////////

  // Optimize for Sigmas for 2 Nodes 1D1P (E= 0.5)
    double sigma_1P = runSigmaSearch(0.001, 1, nCyclesSigma, 3, 1, expected_1P, 1, 1, false);
    double sigma_init_1P = 0.001;

  // Optimize for Sigmas for 2 Nodes 2P2D (E= 3)
    double sigma_2P = runSigmaSearch(0.1, 2.0, nCyclesSigma, 3, 1, expected_2P, 2, 2, true);
    double sigma_init_2P = 0.001;

  ////////////////////////////////////
  // INITIALIZING GRIDSEARCH VALUES //
  ////////////////////////////////////

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

  ////////////
  // TIMING //
  ////////////
    auto t0 = std::chrono::high_resolution_clock::now();

  /////////////////////////
  // RUNNING EXPERIMENTS //
  /////////////////////////

    // 1P1D Gridsearch w/ Brute Force
    runGridSearch(1, 1, false, 1, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, 1, 0.001,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

    // 1P1D Gridsearch w/ Metropolis
    runGridSearch(1, 1, false, 2, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, 1, 0.001,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

    // 1P1D Gridsearch w/ Gibbs
    runGridSearch(1, 1, false, 3, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, sigma_1P, sigma_init_1P,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

    // 2P1D Gridsearch w/ Metropolis
    runGridSearch(2, 1, true, 2, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, 1, 0.001,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

    // 2P2D Gridsearch w/ Metropolis
    runGridSearch(2, 2, true, 2, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, 1, 0.001,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

    // 2P2D Gridsearch w/ Gibbs
    runGridSearch(2, 2, true, 3, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, sigma_2P, sigma_init_2P,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

    // 2P3D Gridsearch w/ Metropolis
    runGridSearch(2, 3, true, 2, nCyclesPow2,
                  whichOptimizer, etaVals, hiddenVals, 1, 0.001,
                  nOptimizeIters, stepLength, timeStep, tolerance, omega);

  ////////////
  // TIMING //
  ////////////
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = (t1 - t0);
    std::cout << "Total Time Elapsed = " << dt.count() << " s\n";

    return 0;
}

double runSigmaSearch(double sigma_min, double sigma_max, int nCyclesPow2, int samplingRule, int whichOptimizer, double expected, int nParticles, int nDims, bool interaction) {
    int nHidden = 2;
    int N_sigmas = 1000;

    double d_sigma = (sigma_max - sigma_min)/(N_sigmas-1);
    double sigmas [N_sigmas] = {};
    for (int i = 0; i < N_sigmas; i++) {
      sigmas[i] = sigma_min + i*d_sigma;
    }

    // SIZE IS HARDCODED!!!! CHECK THE LOOP TO BE SURE ALL IS CONSISTENT
    double meanEnergies [N_sigmas] = { };

    double omega = 1.0; //in hamiltonian
    double sigma_init = 0.00131; //initial spread of initial positions and weights
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
    double minimum = 1E5;
    int min_idx = -1;
    for (int i = 0; i < N_sigmas; i++) {
        if (abs(meanEnergies[i]-3) < minimum) {
            minimum = abs(meanEnergies[i]-expected);
            min_idx = i;
        }
    }
    std::cout << " – BEST SIGMA= " << sigmas[min_idx];
    std::cout << ", dE= " << minimum << std::endl;

    std::string filename = "./Data/sigmas_";
    filename.append(std::to_string(nParticles) + "p_");
    filename.append(std::to_string(nDims) + "d_");
    filename.append(std::to_string(nHidden) + "h_");
    filename.append(std::to_string(nMCcycles) + "cycles_");
    filename.append(std::to_string(samplingRule) + "s_");
    filename.append(std::to_string(eta) + "eta.bin");

    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    outfile.write(reinterpret_cast<const char*> (sigmas), N_sigmas*sizeof(double));
    outfile.close();
    cout << " * Sigmas written to " << filename << endl;

    std::string filename2 = "./Data/sigmas_E_";
    filename2.append(std::to_string(nParticles) + "p_");
    filename2.append(std::to_string(nDims) + "d_");
    filename2.append(std::to_string(nHidden) + "h_");
    filename2.append(std::to_string(nMCcycles) + "cycles_");
    filename2.append(std::to_string(samplingRule) + "s_");
    filename2.append(std::to_string(eta) + "eta.bin");

    std::ofstream outfile2;
    outfile2.open(filename2, std::ios::out | std::ios::binary | std::ios::trunc);
    outfile2.write(reinterpret_cast<const char*> (meanEnergies), N_sigmas*sizeof(double));
    outfile2.close();
    cout << " * Sigma Energies written to " << filename << endl;

    return sigmas[min_idx];
}

double runSigmaInitSearch(double sigma_min, double sigma_max, int nCyclesPow2, int samplingRule, int whichOptimizer, double expected, int nParticles, int nDims, double sigma, bool interaction) {
    int nHidden = 2;
    int N_sigmas = 1000;

    double d_sigma = (sigma_max - sigma_min)/(N_sigmas-1);
    double sigmas [N_sigmas] = {};
    for (int i = 0; i < N_sigmas; i++) {
      sigmas[i] = sigma_min + i*d_sigma;
    }

    // SIZE IS HARDCODED!!!! CHECK THE LOOP TO BE SURE ALL IS CONSISTENT
    double meanEnergies [N_sigmas] = { };

    double omega = 1.0; //in hamiltonian
    if(interaction) {assert(nParticles > 1);}

    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    double eta = 0.1; //learning rate
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 694206661337; //seed does nothing apparently

    int counter = 0;
    std::cout << "OPTIMIZING FOR SIGMA INIT 0%" << std::flush;

    #pragma omp parallel for
    for (int i = 0; i < N_sigmas; i++) {
        NeuralQuantumState nqs(nParticles, nDims, nHidden, sigma, seed, sigmas[i], samplingRule); //must be initialized first
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
        std::cout << "\rOPTIMIZING FOR SIGMA INIT " << round(100*counter/N_sigmas) << "%" << std::flush;
    }
    double minimum = 1E5;
    int min_idx = -1;
    for (int i = 0; i < N_sigmas; i++) {
        if (abs(meanEnergies[i]-3) < minimum) {
            minimum = abs(meanEnergies[i]-expected);
            min_idx = i;
        }
    }
    std::cout << " – BEST SIGMA INIT= " << sigmas[min_idx];
    std::cout << ", dE= " << minimum << std::endl;

    std::string filename = "./Data/sigma_inits_";
    filename.append(std::to_string(nParticles) + "p_");
    filename.append(std::to_string(nDims) + "d_");
    filename.append(std::to_string(nHidden) + "h_");
    filename.append(std::to_string(nMCcycles) + "cycles_");
    filename.append(std::to_string(samplingRule) + "s_");
    filename.append(std::to_string(eta) + "eta.bin");

    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    outfile.write(reinterpret_cast<const char*> (sigmas), N_sigmas*sizeof(double));
    outfile.close();
    cout << " * Sigma_Inits written to " << filename << endl;

    std::string filename2 = "./Data/sigma_inits_E_";
    filename2.append(std::to_string(nParticles) + "p_");
    filename2.append(std::to_string(nDims) + "d_");
    filename2.append(std::to_string(nHidden) + "h_");
    filename2.append(std::to_string(nMCcycles) + "cycles_");
    filename2.append(std::to_string(samplingRule) + "s_");
    filename2.append(std::to_string(eta) + "eta.bin");

    std::ofstream outfile2;
    outfile2.open(filename2, std::ios::out | std::ios::binary | std::ios::trunc);
    outfile2.write(reinterpret_cast<const char*> (meanEnergies), N_sigmas*sizeof(double));
    outfile2.close();
    cout << " * Sigma_Init Energies written to " << filename << endl;

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

void runGridSearch(int nParticles, int nDims, bool interaction, int samplingRule, int nCyclesPow2, int whichOptimizer, std::vector<double> etaVals, std::vector<int> hiddenVals, double sigma, double sigma_init, int nOptimizeIters, double stepLength, double timeStep, double tolerance, double omega){

    auto t0 = std::chrono::high_resolution_clock::now();

    if(interaction) {assert(nParticles > 1);}
    int nMCcycles = (int)pow(2, nCyclesPow2);

    long seed = 1337;
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
            sampler.m_printOptimInfo = false;
            sampler.runOptimization();
            sampler.runDataCollection(nMCcycles*8);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = (t1 - t0);
    std::cout << " – COMPLETE WITH T= " << dt.count() << " s" << std::endl;
}
