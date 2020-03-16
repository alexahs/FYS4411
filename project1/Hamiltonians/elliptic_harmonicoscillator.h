#pragma once
#include "hamiltonian.h"
#include <vector>

class EllipticHarmonicOscillator : public Hamiltonian {
public:
    EllipticHarmonicOscillator(System* system, double omega);
    double computeLocalEnergy(std::vector<Particle*> particles);
    double computeLocalEnergyDerivative(std::vector<class Particle*> particles);

private:
    double m_omega = 0;
};
