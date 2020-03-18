#pragma once
#include "hamiltonian.h"
#include <vector>

class EllipticHarmonicOscillator : public Hamiltonian {
public:
    EllipticHarmonicOscillator(System* system, double gamma);
    double computeLocalEnergy(std::vector<Particle*> particles);
    double computeLocalEnergyDerivative(std::vector<class Particle*> particles);

private:
    double m_gamma2 = 0;
};
