#pragma once
#include "hamiltonian.h"
#include <vector>

class EllipticHarmonicOscillator : public Hamiltonian {
public:
    EllipticHarmonicOscillator(System* system, double gamma, double bosonDiameter);
    double computeLocalEnergy(std::vector<Particle*> particles);
    
private:
    double m_gamma2 = 0;
    double m_bosonDiameter = 0;
};
