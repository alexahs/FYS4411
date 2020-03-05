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

    if (m_system->getNumericalDoubleDerivative()) {
        // Compute the numerical double derivative
        double h = m_system->getStepLength();
        double h2 = h*h;
        double minusalpha = - m_parameters.at(0);
        double dim = particles[0]->getPosition().size();
        // center piece of numerical double derivation
        double waveFunc = this->evaluate(particles);
        double doubleDerivative = -2*dim*waveFunc;
        // NOTE: we will divide by h^2 in the end to reduce # of FLOPS
        double r2 = m_system->getSumRiSquared();
        // The idea is to have the full Sum((r_i)^2), and calculate
        // the difference to obtain r_(i+h,j,k) and so on
        std::vector<double> r2_plus(dim, r2);
        std::vector<double> r2_minus(dim, r2);
        double diff, xi;

        for (auto particle : particles) { // loop over particles
            for (int i=0; i<dim; i++) {
                xi = particle->getPosition()[i];
                diff = xi*xi;
                // Remove this part from the sum
                r2_plus[i] -= diff;
                r2_minus[i] -= diff;
                // Add the shifted position squared
                r2_plus[i] += (xi+h) * (xi+h);
                r2_minus[i] += (xi-h) * (xi-h);
            }
        }
        // Now we have the shifted Sum((r_i)^2) which we will use to evaluate
        // the wave function here:
        for (int i=0; i<dim; i++) {
            doubleDerivative += exp(minusalpha*r2_plus[i]);
            doubleDerivative += exp(minusalpha*r2_minus[i]);
        }
        // Divide by step length squared
        doubleDerivative /= h2;
        return doubleDerivative / waveFunc;
    } else {
        // Use the analytic expression for the double derivative
        double twoAlpha = 2 * m_parameters.at(0);
        double term = 0;
        int dim = particles.at(0)->getPosition().size();
        int N = particles.size();
        double r2 = m_system->getSumRiSquared();
        return -twoAlpha*(dim*N - twoAlpha*r2);
    }
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
    return -m_system->getSumRiSquared();
}
