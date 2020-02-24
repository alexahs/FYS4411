#ifndef SIMPLEGAUSSIAN_H
#define SIMPLEGAUSSIAN_H

#include <vector>
#include "wavefunction.h"

class SimpleGaussian : public WaveFunction
{
public:
    SimpleGaussian(class System* system, double alpha);
    double evaluate(std::vector<class Particle*> particles);
    double computeDoubleDerivative(std::vector<class Particle*> particles);


};

#endif //SIMPLEGAUSSIAN_H
