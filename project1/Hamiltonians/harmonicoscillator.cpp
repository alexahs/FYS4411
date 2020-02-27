#include "harmonicoscillator.h"
#include "hamiltonian.h"
#include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"
#include <cassert>
#include <iostream>

HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
    Hamiltonian(system)
{
    assert(omega > 0);
    m_omega = omega;
}

double HarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles)
{
    /* Here, you need to compute the kinetic and potential energies. Note that
     * when using numerical differentiation, the computation of the kinetic
     * energy becomes the same for all Hamiltonians, and thus the code for
     * doing this should be moved up to the super-class, Hamiltonian.
     *
     * You may access the wave function currently used through the
     * getWaveFunction method in the m_system object in the super-class, i.e.
     * m_system->getWaveFunction()...
     */
     double r2 = 0;
     for (auto particle : particles) {
         for (auto x : particle->getPosition()) {
             r2 += x*x;
         }
     }
    double potentialEnergy = r2*0.5*m_omega*m_omega;
    double kineticEnergy = - 0.5*m_system->getWaveFunction()->computeDoubleDerivative(particles);

    return kineticEnergy + potentialEnergy;
}
