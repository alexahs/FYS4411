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
    m_bosonDiameter2 = bosonDiameter * bosonDiameter;
}

double EllipticHarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles)
{
    double localEnergy = 0, diff, Vint;
    std::vector<double> pos_i(3, 0);
    std::vector<double> pos_j(3, 0);

    for (int i=0; i<particles.size(); i++) {
        localEnergy += m_system->getWaveFunction()->analyticDoubleDerivative(particles, i);
        pos_i = particles[i]->getPosition();
        localEnergy += pos_i[0]*pos_i[0] + pos_i[1]*pos_i[1] + m_gamma2*pos_i[2]*pos_i[2];

        // Repulsive potential
        double distance = 0;
        for (int j=i+1; j<particles.size(); j++) {
            pos_j = particles[j]->getPosition();
            diff = (pos_j[0] - pos_i[0]) * (pos_j[0] - pos_i[0]);
            diff += (pos_j[1] - pos_i[1]) * (pos_j[1] - pos_i[1]);
            diff += (pos_j[2] - pos_i[2]) * (pos_j[2] - pos_i[2]);
            if (diff <= m_bosonDiameter2) {
                Vint += 1000000;
            }
        }
    }

    return localEnergy + Vint;
}


double EllipticHarmonicOscillator::computeLocalEnergyDerivative(std::vector<class Particle*> particles)
{
    return 1;
}
