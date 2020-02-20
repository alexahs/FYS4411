#ifndef SIMPLEGAUSSIAN_H
#define SIMPLEGAUSSIAN_H

class SimpleGaussian : public WaveFunction
{
public:
    SimpleGaussian(double alpha);
    double evaluate(std::vector<class Particle*> particles) = 0;
    double computeDoubleDerivative(std::vector<class Particle*> particles) = 0;


}
