#include "wavefunction.h"
#include <iostream>

WaveFunction::WaveFunction(System* system) {
    m_system = system;
}


double WaveFunction::computeDoubleDerivative(std::vector<class Particle*> particles)
{
   /* Implemented by using the three point formula with a step length.
    *
    * Example (1 dimension):
    *    doubleDerivative = (u_i+1 - 2u_i + u_i-1) / h^2
    *
    * Params: particles
    * Returns: doubleDerivative / waveFunc
    */
    double h = m_system->getStepLength();
    double minus2h = -2*h;
    int dim = particles.at(0)->getPosition().size();
    // center piece of numerical double derivation
    double waveFunc = this->evaluate(particles);
    // The center piece part of the 2nd derivative
    double doubleDerivative = -2*dim*waveFunc;
    // Loop over dimensions
    for (int i=0; i<dim; i++) {
        // Shift all particles +h in one dimension
        for (auto particle : particles) { particle->adjustPosition(h, i); }
        doubleDerivative += this->evaluate(particles);
        // Shift particles back -h and then -h (In total: -2h)
        for (auto particle : particles) { particle->adjustPosition(minus2h, i); }
        doubleDerivative += this->evaluate(particles);
        // Shift particles back to beginning
        for (auto particle : particles) { particle->adjustPosition(h, i); }
    }
    // Divide by step length squared
    return doubleDerivative / (h*h*waveFunc);
}
