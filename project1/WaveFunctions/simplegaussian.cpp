#include "simplegaussian.h"
#include "wavefunction.h"
#include "../particle.h"
#include "../system.h"
#include <cmath>
#include <cassert>


SimpleGaussian::SimpleGaussian(System* system, double alpha) :
    WaveFunction(system)
{
    assert(alpha >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(alpha);
};

double SimpleGaussian::evaluate(std::vector<class Particle*> particles)
{
    /* You need to implement a Gaussian wave function here. The positions of
     * the particles are accessible through the particle[i].getPosition()
     * function.
     *
     * For the actual expression, use exp(-alpha * r^2), with alpha being the
     * (only) variational parameter.
     */


     double wf = 0;

     int N = m_system

     for(int i = 0; i < )


    return 0;
};

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
    return 0;
};
