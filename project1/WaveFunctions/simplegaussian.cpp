#include "simplegaussian.h"
#include <cmath>
#include <cassert>
#include <iostream>


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

double SimpleGaussian::computeDoubleDerivative(std::vector<class Particle*> particles)
{
    /* All wave functions need to implement this function, so you need to
     * find the double derivative analytically. Note that by double derivative,
     * we actually mean the sum of the Laplacians with respect to the
     * coordinates of each particle.
     *
     * This quantity is needed to compute the (local) energy (consider the
     * Schrödinger equation to see how the two are related).
     */
    double twoAlpha = 2 * m_parameters.at(0);
    double term = 0;
    int dim = particles.at(0)->getPosition().size();
    int N = particles.size();
    double r2 = m_system->getSumRiSquared();
    return -twoAlpha*(dim*N - twoAlpha*r2);
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
