#include "standardmetropolistest.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"

#include <cmath>


StandardMetropolisTest::StandardMetropolisTest(System* system) :
        MetropolisTest(system) {
    int a = 1;
}

bool StandardMetropolisTest::metropolisTest()
{
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


    // return true;
}
