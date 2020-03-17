#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "InitialStates/uniformlattice.h"
#include "Math/random.h"
#include "Misc/system.h"
#include "Misc/particle.h"
#include "Misc/sampler.h"
#include "Misc/wfsampler.h"
#include "Misc/writefile.h"

using namespace std;

/*
TODO: (updated 16.03)
    - Finish implementing elliptic HO class
    - Finish implementing correlated WF:
        - evaluation of the WF(done)
        - implement gradiet and local energy
        - implement 2nd derivative
    - Implement gradient descent for correlated WF
    - Implement better particle initialization for correlated WF(done)
    - Implement oneBody density



*/

// Run VMC for spherical HO trap
void run_bruteforce_vmc(double alpha_min,
                        double alpha_max,
                        double alpha_step,
                        int dim,
                        int nParticles,
                        int nCycles,
                        bool numerical);
void run_gradient_descent(int nAlphas, double alpha0, double gamma);
void run_single_vmc(double alpha, int numberOfSteps);
void test_correlated(double alpha, int numberOfSteps, int numberOfParticles);
void vmc_brute_loop();

int main(int argc, char* argv[]) {
    // NOTE: number of metro steps must be a power of 2 for blocking resampling to run

    // if (argc < 2){
    //     cout << "Must provide arguments" << endl;
    //     return 1;
    // }
    //
    // int nParticles = atoi(argv[1]);

    vmc_brute_loop();

    // run_bruteforce_vmc(0.1, 0.9, 0.05);
    // run_gradient_descent(500, 0.2, 0.001);
    // run_single_vmc(0.5, pow(2, 18));
    // test_correlated(0.5, pow(2, 18), nParticles);
    return 0;
}


void vmc_brute_loop(){
    double alpha_min = 0.2;
    double alpha_max = 0.9;
    double alpha_step = 0.1;
    vector<int> vec_nParicles = {1, 10, 100, 500};
    vector<int> vec_dimensions = {1, 2, 3};
    vector<bool> numerical = {false, true};
    int nCycles = (int) pow(2, 21);

    // for(int num = 0; num < numerical.size(); num++){
    //     for(int dim = 0; dim < vec_dimensions.size(); dim++){
    //         for(int nPart = 0; nPart < vec_nParicles.size(); nPart++){
    //             run_bruteforce_vmc(alpha_min,
    //                                alpha_max,
    //                                alpha_step,
    //                                vec_dimensions[dim],
    //                                vec_nParicles[nPart],
    //                                nCycles,
    //                                numerical[num]);
    //         }
    //     }
    // }


    for(auto num : numerical){
        for(auto dim : vec_dimensions){
            for(auto nPart : vec_nParicles){
                run_bruteforce_vmc(alpha_min,
                                   alpha_max,
                                   alpha_step,
                                   dim,
                                   nPart,
                                   nCycles,
                                   num);
            }
        }
    }



}



void test_correlated(double alpha, int numberOfSteps, int numberOfParticles){


    int numberOfDimensions         = 3;         // Dimensions
    // int numberOfParticles          = 10;        // Particales in system
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    double hardSphereRadius        = 1;
    bool importanceSampling        = true;     // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = false;     // Otherwise: use analytical expression for 2nd derivative

    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();


    int nProcs = omp_get_num_procs();
    int stepsPerProc = numberOfSteps/nProcs;
    std::vector<std::vector<double>> tempEnergies;
    std::vector<double> allEnergies;


    // #pragma omp parallel for schedule(dynamic)
        // for(int i=0; i < nProcs; i++){
            System* system = new System();
            system->setSampler                  (new Sampler(system));
            system->setHamiltonian              (new HarmonicOscillator(system, omega));
            system->setWaveFunction             (new SimpleGaussian(system, alpha));
            system->setInitialState             (new UniformLattice(system,
                                                        numberOfDimensions,
                                                        numberOfParticles,
                                                        characteristicLength,
                                                        hardSphereRadius));
            //
            // system->setEquilibrationFraction     (equilibration);
            // system->setStepLength                (stepLength);
            // system->setNumberOfMetropolisSteps   (numberOfSteps);
            // system->setImportanceSampling        (importanceSampling, timeStep);
            // system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
            // system->runMetropolisSteps           ();
            // vector<double> energySamples = system->getSampler()->getEnergySamples();
            // #pragma omp critial
            // tempEnergies.push_back(energySamples);
        // }//end parallel


    // for(int i = 0; i < nProcs; i++){
    //     allEnergies.insert(allEnergies.end(),
    //     tempEnergies[i].begin(),
    //     tempEnergies[i].end());
    // }
    //
    // chrono::steady_clock::time_point end = chrono::steady_clock::now();
    // printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    // writeFileEnergy(allEnergies, numberOfDimensions, numberOfParticles, numberOfSteps);

}

void run_gradient_descent(int nAlphas, double alpha0, double gamma){

    int numberOfDimensions         = 3;         // Dimensions
    int numberOfParticles          = 10;        // Particales in system
    int numberOfSteps              = (int) 1e4; // Monte Carlo cycles
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    bool importanceSampling        = true;     // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = false;     // Otherwise: use analytical expression for 2nd derivative

    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    vector <double> alphaVec;
    vector<double> energyVec;
    vector<double> energy2Vec;
    vector<double> varianceVec;
    vector<double> acceptRatioVec;


    int maxIter = 100;
    double tol = 1e-7;

    double alphaNew = alpha0;
    double alphaOld = 0;
    int iter = 1;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    while (abs(alphaNew - alphaOld) > tol && iter < maxIter){

        System* system = new System();
        system->setSampler                  (new WfSampler(system));
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new SimpleGaussian(system, alphaNew));
        system->setInitialState             (new RandomUniform(system,
                                                    numberOfDimensions,
                                                    numberOfParticles,
                                                    characteristicLength));
        system->setEquilibrationFraction     (equilibration);
        system->setStepLength                (stepLength);
        system->setNumberOfMetropolisSteps   (numberOfSteps);
        system->setImportanceSampling        (importanceSampling, timeStep);
        system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
        system->runMetropolisSteps           ();
        Sampler* system_sampler = system->getSampler();


        // save observables
        alphaVec.push_back(alphaNew);
        energyVec.push_back(system_sampler->getEnergy());
        energy2Vec.push_back(system_sampler->getEnergy2());
        varianceVec.push_back(system_sampler->getVariance());
        acceptRatioVec.push_back(system_sampler->getAcceptRatio());


        // get cost
        double cost = system->getWaveFunction()->evaluateCostFunction();

        alphaOld = alphaNew;

        // compute new alpha with GD
        alphaNew -= gamma*cost;

        iter++;

    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
    printFinal(1, elapsedTime);
    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), numericalDoubleDerviative,
      alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec, elapsedTime);

    if (iter < maxIter){
        cout << " * Converged in " << iter << " steps" << endl;
    }
    else{
        cout << " * Did not converge within tol=" << tol << " in "  << maxIter << " steps" << endl;
    }


}

void run_single_vmc(double alpha, int numberOfSteps){
    int numberOfDimensions         = 3;         // Dimensions
    int numberOfParticles          = 100;        // Particales in system
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    bool importanceSampling        = true;     // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = false;     // Otherwise: use analytical expression for 2nd derivative

    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();


    int nProcs = omp_get_num_procs();
    int stepsPerProc = numberOfSteps/nProcs;
    std::vector<std::vector<double>> tempEnergies;
    std::vector<double> allEnergies;


    #pragma omp parallel for schedule(dynamic)
        for(int i=0; i < nProcs; i++){
            System* system = new System();
            system->setSampler                  (new Sampler(system));
            system->setHamiltonian              (new HarmonicOscillator(system, omega));
            system->setWaveFunction             (new SimpleGaussian(system, alpha));
            system->setInitialState             (new RandomUniform(system,
                                                        numberOfDimensions,
                                                        numberOfParticles,
                                                        characteristicLength));

            system->setEquilibrationFraction     (equilibration);
            system->setStepLength                (stepLength);
            system->setNumberOfMetropolisSteps   (numberOfSteps);
            system->setImportanceSampling        (importanceSampling, timeStep);
            system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
            system->runMetropolisSteps           ();
            vector<double> energySamples = system->getSampler()->getEnergySamples();
            #pragma omp critial
            tempEnergies.push_back(energySamples);
        }//end parallel


    for(int i = 0; i < nProcs; i++){
        allEnergies.insert(allEnergies.end(),
        tempEnergies[i].begin(),
        tempEnergies[i].end());
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    writeFileEnergy(allEnergies, numberOfDimensions, numberOfParticles, numberOfSteps);

}

void run_bruteforce_vmc(double alpha_min,
                        double alpha_max,
                        double alpha_step,
                        int dim,
                        int nParticles,
                        int nCycles,
                        bool numerical) {

    int numberOfDimensions         = dim;         // Dimensions
    int numberOfParticles          = nParticles;        // Particales in system
    int numberOfSteps              = (int) nCycles;  // Monte Carlo cycles
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    bool importanceSampling        = false;     // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = numerical;     // Otherwise: use analytical expression for 2nd derivative
    bool saveEnergySamples         = false;
    // Initialize vectors where results will be stored
    vector<double> alphaVec;
    for(double alpha=alpha_min; alpha<=alpha_max; alpha+=alpha_step) { alphaVec.push_back(alpha); }
    vector<double> energyVec(alphaVec.size(), 0);
    vector<double> energy2Vec(alphaVec.size(), 0);
    vector<double> varianceVec(alphaVec.size(), 0);
    vector<double> acceptRatioVec(alphaVec.size(), 0);
    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<alphaVec.size(); i++) {
            System* system = new System();
            system->setSampler                  (new Sampler(system));
            system->setHamiltonian              (new HarmonicOscillator(system, omega));
            system->setWaveFunction             (new SimpleGaussian(system, alphaVec.at(i)));
            system->setInitialState             (new RandomUniform(system,
                                                        numberOfDimensions,
                                                        numberOfParticles,
                                                        characteristicLength));

            system->setEquilibrationFraction     (equilibration);
            system->setStepLength                (stepLength);
            system->setNumberOfMetropolisSteps   (numberOfSteps);
            system->setImportanceSampling        (importanceSampling, timeStep);
            system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
            system->runMetropolisSteps           ();

            Sampler* system_sampler = system->getSampler();
            energyVec.at(i) = (system_sampler->getEnergy());
            energy2Vec.at(i) = (system_sampler->getEnergy2());
            varianceVec.at(i) = (system_sampler->getVariance());
            acceptRatioVec.at(i) = (system_sampler->getAcceptRatio());
        }
        //end parallel region

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
    printFinal(1, elapsedTime);
    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), numericalDoubleDerviative,
      alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec, elapsedTime);
}
