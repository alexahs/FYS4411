#include "simplegaussian.h"
#include <cmath>
#include <cassert>
<<<<<<< HEAD
=======
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"
>>>>>>> 40c8220fb37c7d7c850a1babb27cf0438dbdc38f

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
    double waveFunc = 1;
    double alpha = m_parameters.at(0);

    for (auto particle : particles) {
        double r2 = 0;
        for (auto x : particle->getPosition()) {
            r2 += x*x;
        }
        waveFunc *= exp(- alpha*r2);
    }

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
    for (auto particle : particles) { // loop over particles
        double r2 = 0;
        for (auto xi : particle->getPosition()) { // loop over dimensions
            r2 += xi*xi;
        }
        term += 1 - twoAlpha*r2;
    }

    return -twoAlpha*term;
}
