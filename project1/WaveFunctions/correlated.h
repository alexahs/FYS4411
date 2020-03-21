#pragma once
#include "wavefunction.h"

class Correlated : public WaveFunction {
public:
    Correlated(class System* system, double alpha, double beta, double bosonDiameter);
    double evaluate(std::vector<class Particle*> particles);
    double analyticDoubleDerivative(std::vector<class Particle*> particles, int k);
    std::vector<double> computeQuantumForce(class Particle* particle);
    double evaluateDerivative(std::vector<class Particle*> particles);
    double evaluateCostFunction();

private:
    double computeSingleOneBodyPart(Particle* particle);
    double computeSingleDistance(Particle* p1, Particle* p2);
    double computeFullOneBodyPart(std::vector<class Particle*> particles);
    double computeSingleInteractingPart(Particle* p1, Particle* p2);
    double dotProduct(std::vector<double> v1, std::vector<double> v2);
    double m_bosonDiameter = 0.00433; // Fixed as in refs. (see report)
};
