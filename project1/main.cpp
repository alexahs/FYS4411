#include <iostream>
#include <vector>
#include <chrono>

#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "Math/random.h"
#include "Misc/system.h"
#include "Misc/particle.h"
#include "Misc/sampler.h"
#include "Misc/wfsampler.h"
#include "Misc/writefile.h"

using namespace std;

/*
TODO:
    - Add more generalized wavefunction and hamiltonian classes
    - Add functionality for writing to file (complete for 1 variational parameter)
    - Add importance sampling. (done)
    - Implement gradient descent (done)
    - Implement functionality for resampling
        - bootstrap
        - blocking
        - (this involves appending cumulative energy to an array for each MC step)

*/

// Run VMC for spherical HO trap
void run_bruteforce_vmc(double alpha_min, double alpha_max, double alpha_step);
void run_gradient_descent(int nAlphas, double alpha0, double gamma);
void run_single_vmc(double alpha, int numberOfSteps);

int main() {
    // run_bruteforce_vmc(0.1, 0.9, 0.05);
    // run_gradient_descent(500, 0.2, 0.001);
    run_single_vmc(0.4, 1e6);
    return 0;
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
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), numericalDoubleDerviative,
      alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec);

    if (iter < maxIter){
        cout << " * Converged in " << iter << " steps" << endl;
    }
    else{
        cout << " * Did not converge within tol=" << tol << " in "  << maxIter << " steps" << endl;
    }


}

void run_single_vmc(double alpha, int numberOfSteps){
    int numberOfDimensions         = 3;         // Dimensions
    int numberOfParticles          = 10;        // Particales in system
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

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    writeFileEnergy(energySamples, numberOfDimensions, numberOfParticles, numberOfSteps);

}

void run_bruteforce_vmc(double alpha_min, double alpha_max, double alpha_step) {

    int numberOfDimensions         = 3;         // Dimensions
    int numberOfParticles          = 10;        // Particales in system
    int numberOfSteps              = (int) 1e6; // Monte Carlo cycles
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    bool importanceSampling        = true;     // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = false;     // Otherwise: use analytical expression for 2nd derivative
    bool saveEnergySamples         = true;
    // Initialize vectors where results will be stored
    vector<double> alphaVec;
    for(double alpha=alpha_min; alpha<=alpha_max; alpha+=alpha_step) { alphaVec.push_back(alpha); }
    vector<double> energyVec(alphaVec.size(), 0);
    vector<double> energy2Vec(alphaVec.size(), 0);
    vector<double> varianceVec(alphaVec.size(), 0);
    vector<double> acceptRatioVec(alphaVec.size(), 0);
    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    // #pragma omp parallel for schedule(dynamic)
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
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), numericalDoubleDerviative,
      alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec);
}
