#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <vector>
#include <stdlib>



class WaveFunction
{
public:
    // WaveFunction(class System* system);
    int getNumberOfParameters() {return m_numberOfParameters;}
    std::vector<double> getParameters() {return m_parameters;}
    virtual double evaluate(std::vector<class Particle*> particles);
    virtual double computeDoubleDerivative(std::vector<class Particle*> particles);

protected:
    int m_numberOfParticles = 0;
    std:vector<double> m_parameters = std:vector<double>();
    // class System* m_system = nullptr;
}


#endif //WAVEFUNCTION_H
