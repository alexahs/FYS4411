#ifndef HARMONICOSCILLATOR_H
#define HARMONICOSCILLATOR_H

#include <vector>
#include "hamiltonian.h"
// #include "../particle.h"

class HarmonicOscillator : public Hamiltonian
{
public:
    HarmonicOscillator(System* system, double omega);
    double computeLocalEnergy(std::vector<Particle*> particles);

private:
    double m_omega = 0;
};



#endif //HARMONICOSCILLATOR_H
