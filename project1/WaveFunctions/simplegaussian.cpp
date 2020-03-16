#include "simplegaussian.h"
#include "Misc/wfsampler.h"
#include <cmath>
#include <cassert>
#include <iostream>

using namespace std;

SimpleGaussian::SimpleGaussian(System* system, double alpha) :
        WaveFunction(system) {
    assert(alpha >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(alpha);
}

double SimpleGaussian::evaluate(std::vector<class Particle*> particles)
{
    /* You need to implement a Gaussian wave function here. The positions of
     * the particles are accessible through the particle[i].getPosition()
     * function.
     *
     * For the actual expression, use exp(- alpha * r^2), with alpha being the
     * (only) variational parameter.
     */
    double alpha = m_parameters.at(0);
    double waveFunc = exp(- alpha*m_system->getSumRiSquared());
    return waveFunc;
}

double SimpleGaussian::analyticDoubleDerivative(std::vector<class Particle*> particles)
{
    /* The analytic expression for the doubleDerivative. This function is
     * meant to be overwritten in classes that inherit from the wavefunction
     * class. The purpose is to have an expression which is computationally
     * more efficient than the standard numerical double derivative formula.
     * Params: particles
     * Returns: doubleDerivative / waveFunc
     */
    double twoAlpha = 2 * m_parameters.at(0);
    int dim = particles.at(0)->getPosition().size();
    int num = particles.size();
    double r2 = m_system->getSumRiSquared();
    return twoAlpha*(twoAlpha*r2 - dim*num);
}

std::vector<double> SimpleGaussian::computeQuantumForce(Particle* particle)
{
    int dims = m_system->getNumberOfDimensions();
    double fourAlpha = 4.0*m_parameters.at(0);
    std::vector<double> qForce =  std::vector<double>();
    std::vector<double> pos = particle->getPosition();
    for (int dim = 0; dim < dims; dim++){
        qForce.push_back(-fourAlpha*pos.at(dim));
        // std::cout << qForce.at(dim) << std::endl;
    }

    return qForce;
}

double SimpleGaussian::evaluateDerivative(std::vector<class Particle*> particles)
{   /*
    dWf/dAlpha = -sum{r_i^2}*exp(-alpha*sum{r_i^2})
    returns dW/dAlpha * 1/wf = -sum{r_i^2}
    */
    return -1*m_system->getSumRiSquared();
}


double SimpleGaussian::evaluateCostFunction(){



    double term1 = m_system->getSampler()->getExpectWfDerivTimesLocalE();
    double term2 = m_system->getSampler()->getExpectWfDerivExpectLocalE();

    double cost = 2*(term1/term2 - 1);
    cost *= term2;



    return cost;
}
