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
    /* We have two options in this case. When the wavefunction is defined
     * it takes the variable 'numericalDoubleDerivative', if that variable
     * is true,
     *      Use the three point formula to compute the Laplacian
     * else (default),
     *      Use the analytic expressino for the laplacian.
     */
    double r2 = m_system->getSumRiSquared();
    double potentialEnergy = 0.5*r2*m_omega*m_omega;
    double secondDerivative;
    if (m_system->getNumericalDoubleDerivative()) {
        secondDerivative = m_system->getWaveFunction()->computeDoubleDerivative(particles);
    } else {
        secondDerivative = m_system->getWaveFunction()->analyticDoubleDerivative(particles, 0);
    }
    double kineticEnergy = - 0.5*secondDerivative;

    return kineticEnergy + potentialEnergy;
}
