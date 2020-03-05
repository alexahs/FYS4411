#pragma once
#include "wavefunction.h"
#include <vector>

class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(class System* system, double alpha);
    double evaluate(std::vector<class Particle*> particles);
    double computeDoubleDerivative(std::vector<class Particle*> particles);
    std::vector<double> computeQuantumForce(class Particle* particle);
    double evaluateDerivative(std::vector<class Particle*> particles); 
};
