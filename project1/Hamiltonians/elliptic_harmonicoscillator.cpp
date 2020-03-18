#include "elliptic_harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include "Misc/system.h"
#include "Misc/particle.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/correlated.h"


using std::cout;
using std::endl;


EllipticHarmonicOscillator::EllipticHarmonicOscillator(System* system, double gamma) :
        Hamiltonian(system) {
    /*
    remember to use the scaled version of omega in this potential
    gamma = omega_z/omega_ho
    */
    assert(gamma > 0);
    m_gamma2  = gamma * gamma;
}

double EllipticHarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles)
{
    double localEnergy = 0;
    std::vector<double> pos(3, 0);
    for (int i=0; i<particles.size(); i++) {
        localEnergy += m_system->getWaveFunction()->analyticDoubleDerivative(particles, i);
        pos = particles[i]->getPosition();
        localEnergy += pos[0]*pos[0] + pos[1]*pos[1] + m_gamma2*pos[2]*pos[2];
    }
    return localEnergy;
}


double EllipticHarmonicOscillator::computeLocalEnergyDerivative(std::vector<class Particle*> particles)
{
    return 1;
}
