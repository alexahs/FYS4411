#include "harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include "Misc/system.h"
#include "Misc/particle.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;


HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
        Hamiltonian(system) {
    assert(omega > 0);
    m_omega  = omega;
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
    double r2 = m_system->getSumRiSquared();
    double potentialEnergy = 0.5*r2*m_omega*m_omega;
    double secondDerivative = m_system->getWaveFunction()->computeDoubleDerivative(particles);
    double kineticEnergy = - 0.5*secondDerivative;

    return kineticEnergy + potentialEnergy;
}


double HarmonicOscillator::computeLocalEnergyDerivative(std::vector<class Particle*> particles)
{
    double sumRi2 = m_system->getSumRiSquared();
    double alpha = m_system->getWaveFunction()->getParameters().at(0);
    int N =  m_system->getNumberOfParticles();
    int dims = m_system->getNumberOfDimensions();

    return N*dims - 4*alpha*sumRi2;
}
