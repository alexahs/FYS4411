#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <omp.h>

#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "sampler.h"
#include "Math/random.h"

using std::vector;
using std::cout;
using std::endl;

void runGridSearch1p1d();
void runSingle();
void runGridSearch2();
void runGridSearch3();
void runGridSearch4();

int main(){
    // runSingle();
    // runGridSearch1p1d();
    runGridSearch2();
    runGridSearch3();
    runGridSearch4();
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
    double sigma_init = 1.0; //initial spread of initial positions and weights
    bool interaction = false;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 16;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    // std::vector<double> etaVals {0.25, 0.1, 0.01, 0.001, 0.0001};
    // std::vector<double> etaVals {0.2,0.1895,0.179,0.1684,0.1579,0.1474,0.1369,0.1264,0.1158,0.1053,0.0948,0.0843,0.0737,0.0632,0.0527};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    std::vector<double> etaVals {0.2000,0.1968,0.1937,0.1905,0.1873,0.1842,0.1810,0.1779,0.1747,0.1715,0.1684,0.1652,0.1620,0.1589,0.1557,0.1526,0.1494,0.1462,0.1431,0.1399,0.1367,0.1336,0.1304,0.1272,0.1241,0.1209,0.1178,0.1146,0.1114,0.1083,0.1051,0.1019,0.0988,0.0956,0.0924,0.0893,0.0861,0.0830,0.0798,0.0766,0.0735,0.0703,0.0671,0.0640,0.0608,0.0577,0.0545,0.0513,0.0482,0.0450};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;
}

void runGridSearch2(){

    int nParticles = 2;
    int nDims = 1;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 1.0; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 15;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    // std::vector<double> etaVals {0.25, 0.1, 0.01, 0.001, 0.0001};
    // std::vector<double> etaVals {0.2,0.1895,0.179,0.1684,0.1579,0.1474,0.1369,0.1264,0.1158,0.1053,0.0948,0.0843,0.0737,0.0632,0.0527};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    std::vector<double> etaVals {0.2000,0.1968,0.1937,0.1905,0.1873,0.1842,0.1810,0.1779,0.1747,0.1715,0.1684,0.1652,0.1620,0.1589,0.1557,0.1526,0.1494,0.1462,0.1431,0.1399,0.1367,0.1336,0.1304,0.1272,0.1241,0.1209,0.1178,0.1146,0.1114,0.1083,0.1051,0.1019,0.0988,0.0956,0.0924,0.0893,0.0861,0.0830,0.0798,0.0766,0.0735,0.0703,0.0671,0.0640,0.0608,0.0577,0.0545,0.0513,0.0482,0.0450};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;
}

void runGridSearch3(){

    int nParticles = 2;
    int nDims = 2;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 1.0; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 15;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    // std::vector<double> etaVals {0.25, 0.1, 0.01, 0.001, 0.0001};
    // std::vector<double> etaVals {0.2,0.1895,0.179,0.1684,0.1579,0.1474,0.1369,0.1264,0.1158,0.1053,0.0948,0.0843,0.0737,0.0632,0.0527};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    std::vector<double> etaVals {0.2000,0.1968,0.1937,0.1905,0.1873,0.1842,0.1810,0.1779,0.1747,0.1715,0.1684,0.1652,0.1620,0.1589,0.1557,0.1526,0.1494,0.1462,0.1431,0.1399,0.1367,0.1336,0.1304,0.1272,0.1241,0.1209,0.1178,0.1146,0.1114,0.1083,0.1051,0.1019,0.0988,0.0956,0.0924,0.0893,0.0861,0.0830,0.0798,0.0766,0.0735,0.0703,0.0671,0.0640,0.0608,0.0577,0.0545,0.0513,0.0482,0.0450};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;

}

void runGridSearch4(){

    int nParticles = 2;
    int nDims = 3;
    double sigma = 1.0; //in nqs
    double omega = 1.0; //in hamiltonian
    double sigma_init = 1.0; //initial spread of initial positions and weights
    bool interaction = true;
    if(interaction) {assert(nParticles > 1);}

    int nCyclesPow2 = 15;
    int nMCcycles = (int)pow(2, nCyclesPow2); //number of montecarlo cycles
    int nOptimizeIters = 200; //max iters in optimization
    double stepLength = 0.1; //for standard metropolis stampling
    double timeStep = 0.45; //for importance sampling
    int samplingRule = 2; //1 - standard, 2 - metropolis, 3- gibbs
    int whichOptimizer = 1; //1 - gradient descent, 2 - some other optim scheme
    double tolerance = 1e-6; //tolerance for convergence
    long seed = 1337; //seed does nothing apparently


    // std::vector<double> etaVals {0.25, 0.1, 0.01, 0.001, 0.0001};
    // std::vector<double> etaVals {0.2,0.1895,0.179,0.1684,0.1579,0.1474,0.1369,0.1264,0.1158,0.1053,0.0948,0.0843,0.0737,0.0632,0.0527};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    std::vector<double> etaVals {0.2000,0.1968,0.1937,0.1905,0.1873,0.1842,0.1810,0.1779,0.1747,0.1715,0.1684,0.1652,0.1620,0.1589,0.1557,0.1526,0.1494,0.1462,0.1431,0.1399,0.1367,0.1336,0.1304,0.1272,0.1241,0.1209,0.1178,0.1146,0.1114,0.1083,0.1051,0.1019,0.0988,0.0956,0.0924,0.0893,0.0861,0.0830,0.0798,0.0766,0.0735,0.0703,0.0671,0.0640,0.0608,0.0577,0.0545,0.0513,0.0482,0.0450};//,0.0422,0.0317,0.0211,0.0106,0.0001};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9};
    // std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::vector<int> hiddenVals {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
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
            // sampler.printGridSearchInfo(i, j);
            counter++;
            std::cout << "\rLOADING " << round(100*counter/tot_iter) << "%" << std::flush;

        }
    }
    std::cout << std::endl;

}
