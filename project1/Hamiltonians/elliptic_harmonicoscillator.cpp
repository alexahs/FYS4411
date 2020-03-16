#include "elliptic_harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include "Misc/system.h"
#include "Misc/particle.h"
#include "WaveFunctions/wavefunction.h"


using std::cout;
using std::endl;


EllipticHarmonicOscillator::EllipticHarmonicOscillator(System* system, double omega) :
        Hamiltonian(system) {
    /*
    remember to use the scaled version of omega in this potential
    omega = omega_z/omega_ho
    */
    assert(omega > 0);
    m_omega  = omega;
}

double EllipticHarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles)
{
    return 1;
}


double EllipticHarmonicOscillator::computeLocalEnergyDerivative(std::vector<class Particle*> particles)
{
    return 1;
}
