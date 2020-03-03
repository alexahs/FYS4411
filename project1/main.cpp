#include <iostream>
#include <vector>

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
    - Fix randomuniform.cpp such that it initializes particle positions randomly
      within the bounds of the external potential. (completed)
    - Add more generalized wavefunction and hamiltonian classes
    - Add functionality for writing to file (complete for 1 variational parameter)
    - Add importance sampling. (in progress)
    - Add functionality within system for equilibriating the system
      before sampling begins. (completed)
    - Add counting of accepted steps. (completed)
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
    int numberOfSteps           = (int) 1e5;
    double omega                = 1.0;          // Oscillator frequency.
    double stepLength           = 0.1;         // Metropolis step length.
    double equilibration        = 0.05;          // Amount of the total steps used for equilibration.
    double characteristicLength = 1.0;
    bool importanceSampling     = false;

    double alpha = alpha_min;
    int numAlphas = int((alpha_max - alpha_min)/alpha_step) + 1;
    vector<double> alphaVec;
    vector<double> energyVec;
    vector<double> energy2Vec;
    vector<double> varianceVec;
    vector<double> acceptRatioVec;
    vector<double> sumRiSquaredVec;


    for (int i=0; i<numAlphas; i++) {
        System* system = new System();
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new SimpleGaussian(system, alpha));
        system->setInitialState             (new RandomUniform(system,
                                                    numberOfDimensions,
                                                    numberOfParticles,
                                                    characteristicLength));

        system->setEquilibrationFraction    (equilibration);
        system->setStepLength               (stepLength);
        system->setImportanceSampling       (importanceSampling);
        system->runMetropolisSteps          (numberOfSteps);
        Sampler* system_sampler = system->getSampler();
        alphaVec.push_back(alpha);
        energyVec.push_back(system_sampler->getEnergy());
        energy2Vec.push_back(system_sampler->getEnergy2());
        varianceVec.push_back(system_sampler->getVariance());
        acceptRatioVec.push_back(system_sampler->getAcceptRatio());
        sumRiSquaredVec.push_back(system->getSumRiSquared());
        alpha += alpha_step;
    }
    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec, sumRiSquaredVec);
}
