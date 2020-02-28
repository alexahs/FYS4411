#include <iostream>
#include "system.h"
#include "particle.h"
#include "sampler.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "Math/random.h"

using namespace std;

void run_vmc(double alpha_min, double alpha_max, double alpha_step);

int main() {

    double alpha_min = 0.1;
    double alpha_max = 1.0;
    double alpha_step = 0.1;

    run_vmc(alpha_min, alpha_max, alpha_step);

    return 0;
}


void run_vmc(double alpha_min, double alpha_max, double alpha_step){

    int numberOfDimensions  = 1;
    int numberOfParticles   = 1;
    int numberOfSteps       = (int) 1e5;
    double omega            = 1.0;          // Oscillator frequency.
    double stepLength       = 0.1;          // Metropolis step length.
    double equilibration    = 0.0;          // Amount of the total steps used for equilibration.

    double alpha = alpha_min;

    while (alpha <= alpha_max) {
        System* system = new System();
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new SimpleGaussian(system, alpha));
        system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
        system->setEquilibrationFraction    (equilibration);
        system->setStepLength               (stepLength);
        system->runMetropolisSteps          (numberOfSteps);
        alpha += alpha_step;
    }

}
