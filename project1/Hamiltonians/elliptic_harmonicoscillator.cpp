#include "elliptic_harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include "Misc/system.h"
#include "Misc/particle.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/correlated.h"


using std::cout;
using std::endl;


EllipticHarmonicOscillator::EllipticHarmonicOscillator(System* system, double gamma,
        double bosonDiameter) : Hamiltonian(system) {
    /*
    remember to use the scaled version of omega in this potential
    gamma = omega_z/omega_ho
    */
    assert(gamma > 0);
    m_gamma2  = gamma * gamma;
    m_bosonDiameter = bosonDiameter;
}

double EllipticHarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles)
{
    double localEnergy=0, rkj;
    std::vector<double> rk(3, 0);
    std::vector<double> rj(3, 0);
    // Loop over all particles
    for (int k=0; k<particles.size(); k++) {
        localEnergy -= 0.5*m_system->getWaveFunction()->analyticDoubleDerivative(particles, k);
        rk = particles[k]->getPosition();
        localEnergy += 0.5 * (rk[0]*rk[0] + rk[1]*rk[1] + m_gamma2*rk[2]*rk[2]);

        // Repulsive potential
        for (int j=k+1; j<particles.size(); j++) {
            rj = particles[j]->getPosition();
            rkj =  (rj[0] - rk[0]) * (rj[0] - rk[0]);
            rkj += (rj[1] - rk[1]) * (rj[1] - rk[1]);
            rkj += (rj[2] - rk[2]) * (rj[2] - rk[2]);
            if (sqrt(rkj) <= m_bosonDiameter) {
                localEnergy += 10000;
                cout << "Particle " << k << " and " << j << " has crashed!\n";
            }
        }
    }
    return localEnergy;
}


double EllipticHarmonicOscillator::computeLocalEnergyDerivative(std::vector<class Particle*> particles)
{
    return 1;
}
