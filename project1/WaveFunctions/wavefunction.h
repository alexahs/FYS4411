#pragma once
#include <vector>
#include "Misc/particle.h"
#include "Misc/system.h"

class WaveFunction {
public:
    WaveFunction(class System* system);
    int getNumberOfParameters()                 { return m_numberOfParameters; }
    std::vector<double> getParameters()         { return m_parameters; }
    virtual double evaluate(std::vector<class Particle*> particles) = 0;
    virtual double computeDoubleDerivative(std::vector<class Particle*> particles) = 0;
    virtual std::vector<double> computeQuantumForce(class Particle* particle) = 0;
    virtual double evaluateDerivative(std::vector<class Particle*> particles) = 0; //wrt var. params

protected:
    int     m_numberOfParameters = 0;
    std::vector<double> m_parameters = std::vector<double>();
    System* m_system = nullptr;

};
