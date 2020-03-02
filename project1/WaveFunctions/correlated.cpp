#include "correlated.h"
#include <cmath>
#include <cassert>


Correlated::Correlated(System* system, double alpha, double beta) :
      WaveFunction(system) {
    assert(alpha >= 0);
    assert(beta >= 0);
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
}

double Correlated::evaluate(std::vector<class Particle*> particles) {

    return 1.0;
}


double Correlated::computeDoubleDerivative(std::vector<class Particle*> particles) {


    return 1.0;
}
