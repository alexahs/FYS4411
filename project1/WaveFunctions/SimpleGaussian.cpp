#include "SimpleGaussian.h"
#include "WaveFunction.h"
#include "../particle.h"
#include "../system.h"
#include <cmath>
#include <cassert>


SimpleGaussian::SimpleGaussian(double alpha;)
{
    assert(alpha >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(alpha);
}

double SimpleGaussian::evaluate(std::vector<class Particle*> particles)
{
    //evaluate the wavefunction
    return 1.0
}

double SimpleGaussian::computeDoubleDerivative(std::vector<class Particle*> particles)
{
    return 1.0
}
