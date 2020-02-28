#pragma once
#include "wavefunction.h"

class Correlated : public WaveFunction {
public:
    Correlated(class System* system, double alpha, double beta);
    double evaluate(std::vector<class Particle*> particles);
    double computeDoubleDerivative(std::vector<class Particle*> particles);
};
