#pragma once
#include "wavefunction.h"

class Correlated : public WaveFunction {
public:
    Correlated(class System* system, double alpha, double beta, double radius);
    double evaluate(std::vector<class Particle*> particles);
    double computeDoubleDerivative(std::vector<class Particle*> particles);
    double evaluateCostFunction();

private:
    double computeSingleOneBodyPart(Particle* particle);
    double computeFullOneBodyPart(std::vector<class Particle*> particles);
    double computeSingleInteractingPart(Particle* p1, Particle* p2);
    double computeSingleDistance(Particle* p1, Particle* p2);
    double computeLaplacian(std::vector<class Particle*> particles);
    double m_hardShpereRadius = 1;
};
