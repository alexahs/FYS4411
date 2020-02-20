#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include "../particle.h"

#include <vector>


// using namespace std;

class WaveFunction
{
public:
    // WaveFunction(class System* system);
    int getNumberOfParameters() {return m_numberOfParameters;}
    std::vector<double> getParameters() {return m_parameters;}
    virtual double evaluate(std::vector<class Particle*> particles) = 0;
    virtual double computeDoubleDerivative(std::vector<class Particle*> particles) = 0;

protected:
    int m_numberOfParameters = 1;
    std::vector<double> m_parameters = std::vector<double>();
    // class System* m_system = nullptr;
};


#endif //WAVEFUNCTION_H
