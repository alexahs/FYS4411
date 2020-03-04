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
#include "Misc/writefile.h"

using namespace std;

/*
TODO:
    - Add more generalized wavefunction and hamiltonian classes
    - Add functionality for writing to file (complete for 1 variational parameter)
    - Add importance sampling. (in progress)
    - Implement gradient descent
    - Implement functionality for resampling
        - bootstrap
        - blocking
        - (this involves appending cumulative energy to an array for each MC step)

*/

// Run VMC for spherical HO trap
void run_vmc(double alpha_min, double alpha_max, double alpha_step);

int main() {
    run_vmc(0.1, 1.0, 0.05);
    return 0;
}


void run_vmc(double alpha_min, double alpha_max, double alpha_step) {

    int numberOfDimensions      = 3;
    int numberOfParticles       = 10;
    int numberOfSteps           = (int) 1e6;
    double omega                = 1.0;    // Oscillator frequency.
    double stepLength           = 0.1;   // Metropolis: step length
    double timeStep             = 0.01;   // Metropolis-Hastings: time step
    double h                    = 0.001;   // Double derivative step length
    double equilibration        = 0.05;   // Amount of the total steps used for equilibration.
    double characteristicLength = 1.0;
    bool importanceSampling     = false;
    bool numericalDoubleDerviative = true;
    // int numAlphas = int((alpha_max - alpha_min)/alpha_step) + 1;
    vector<double> alphaVec;
    vector<double> energyVec;
    vector<double> energy2Vec;
    vector<double> varianceVec;
    vector<double> acceptRatioVec;
    // auto start = high_resolution_clock::now();
    for(double alpha = alpha_min; alpha <= alpha_max; alpha += alpha_step){
        alphaVec.push_back(alpha);
    }

    // #pragma omp parallel
    #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < alphaVec.size(); i++){
        // for(auto alpha : alphaVec){
            System* system = new System();
            system->setHamiltonian              (new HarmonicOscillator(system, omega));
            system->setWaveFunction             (new SimpleGaussian(system, alphaVec.at(i)));
            system->setInitialState             (new RandomUniform(system,
                                                        numberOfDimensions,
                                                        numberOfParticles,
                                                        characteristicLength));

            system->setEquilibrationFraction     (equilibration);
            system->setStepLength                (stepLength);
            system->setStepLength                (stepLength);
            system->setStepLength                (stepLength);
            system->setImportanceSampling        (importanceSampling, timeStep);
            system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
            system->runMetropolisSteps           (numberOfSteps);
            Sampler* system_sampler = system->getSampler();
            // alphaVec.push_back(alpha);
            energyVec.push_back(system_sampler->getEnergy());
            energy2Vec.push_back(system_sampler->getEnergy2());
            varianceVec.push_back(system_sampler->getVariance());
            acceptRatioVec.push_back(system_sampler->getAcceptRatio());
            // alpha += alpha_step;
        }
        //end omp

    // auto stop = high_resolution_clock::now()

    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), numericalDoubleDerviative,
      alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec);
}
