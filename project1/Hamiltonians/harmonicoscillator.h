#ifndef HARMONICOSCILLATOR_H
#define HARMONICOSCILLATOR_H

#include <vector>
#include "hamiltonian.h"

class HarmonicOscillator : public Hamiltonian
{
public:
    HarmonicOscillator(double omega);
    double computeLocalEnergy(std::vector<Particle*> particles);

private:
    double m_omega = 0;
};



#endif //HARMONICOSCILLATOR_H
