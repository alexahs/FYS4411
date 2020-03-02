#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include <cassert>
#include <iostream>

using std::cout;
using std::endl;

bool System::metropolisStep() {
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change it's position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated
     * at this new position with the one at the old position).
     */
    int rnd_idx = Random::nextInt(m_numberOfParticles);
    Particle* random_particle = m_particles.at(rnd_idx);
    double wf_old = m_waveFunction->evaluate(m_particles);
    std::vector<double> proposed_steps = std::vector<double>();

    for (int dim=0; dim<m_numberOfDimensions; dim++){
        proposed_steps.push_back(m_stepLength*(Random::nextDouble() - 0.5));
        random_particle->adjustPosition(proposed_steps.at(dim), dim); // Adjust position
    }
    double wf_new = m_waveFunction->evaluate(m_particles); // Evaluate new wave function

    if (Random::nextDouble()*wf_old*wf_old <= wf_new*wf_new) {
        return true;
    } else {
        // If move is rejected, then revert to the old position
        for (int dim=0; dim<m_numberOfDimensions; dim++) {
            random_particle->adjustPosition(- proposed_steps.at(dim), dim);
        }
        return false;
    }
}

void System::runMetropolisSteps(int numberOfMetropolisSteps) {
    m_particles                 = m_initialState->getParticles();
    m_sampler                   = new Sampler(this);
    m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
    m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);

    // Equilibriation
    for (int i=0; i<numberOfMetropolisSteps*m_equilibrationFraction; i++) {
        bool acceptedEquilibriateStep = metropolisStep();
    }

    for (int i=0; i<numberOfMetropolisSteps; i++) {
        bool acceptedStep = metropolisStep();

        /* Here you should sample the energy (and maybe other things using
         * the m_sampler instance of the Sampler class. Make sure, though,
         * to only begin sampling after you have let the system equilibrate
         * for a while. You may handle this using the fraction of steps which
         * are equilibration steps; m_equilibrationFraction.
         */
        m_sampler->sample(acceptedStep);
    }
    m_sampler->computeAverages();
    m_sampler->printOutputToTerminal();
}

void System::setNumberOfParticles(int numberOfParticles) {
    m_numberOfParticles = numberOfParticles;
}

void System::setNumberOfDimensions(int numberOfDimensions) {
    m_numberOfDimensions = numberOfDimensions;
}

void System::setStepLength(double stepLength) {
    assert(stepLength >= 0);
    m_stepLength = stepLength;
}

void System::setEquilibrationFraction(double equilibrationFraction) {
    assert(equilibrationFraction >= 0);
    m_equilibrationFraction = equilibrationFraction;
}

void System::setHamiltonian(Hamiltonian* hamiltonian) {
    m_hamiltonian = hamiltonian;
}

void System::setWaveFunction(WaveFunction* waveFunction) {
    m_waveFunction = waveFunction;
}

void System::setInitialState(InitialState* initialState) {
    m_initialState = initialState;
}

double System::getSumRiSquared() {
    double r2 = 0;
    for (auto particle : m_particles) { // loop over particles
        for (auto xi : particle->getPosition()) { // loop over dimensions
            r2 += xi*xi;
        }
    }
    return r2;
}
