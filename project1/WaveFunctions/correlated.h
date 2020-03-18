#pragma once
#include "wavefunction.h"

class Correlated : public WaveFunction {
public:
    Correlated(class System* system, double alpha, double beta);
    double evaluate(std::vector<class Particle*> particles);
    double analyticDoubleDerivative(std::vector<class Particle*> particles, int k);
    std::vector<double> computeQuantumForce(class Particle* particle);
    double evaluateDerivative(std::vector<class Particle*> particles);
    double evaluateCostFunction();

private:
    double computeSingleOneBodyPart(Particle* particle);
    double computeFullOneBodyPart(std::vector<class Particle*> particles);
    double computeSingleInteractingPart(Particle* p1, Particle* p2);
    double computeSingleDistance(Particle* p1, Particle* p2);
    double m_bosonDiameter = 0.0043; // Fixed as in refs. (see report)
};
