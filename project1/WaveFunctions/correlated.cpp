#include "correlated.h"
#include "Misc/wfsampler.h"
#include <cmath>
#include <cassert>

#include <iostream>
using namespace std;

Correlated::Correlated(System* system, double alpha, double beta, double bosonDiameter) :
      WaveFunction(system) {
    assert(alpha >= 0);
    assert(beta >= 0);
    assert(system->getNumberOfDimensions() == 3);
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_bosonDiameter = bosonDiameter;
}

double Correlated::evaluate(std::vector<class Particle*> particles) {
    /*
    Evaluate the wave function given the position of all particles.
    */
    // Elliptical Gaussian contribution from each particle
    double wf = computeFullOneBodyPart(particles);
    // Correlation contribution
    int num = m_system->getNumberOfParticles();
    for (int i=0; i<num; i++) {
        for (int j=i+1; j<num; j++) {
            wf *= computeSingleInteractingPart(particles[i], particles[j]);
        }
    }
    return wf;
}

double Correlated::analyticDoubleDerivative(std::vector<class Particle*> particles, int k) {
    /*
    Actually computes the (Laplacian / wave function). This is the
    analytic expression for the fully correlated wave function.
    */
    std::vector<double> rk = particles[k]->getPosition();
    double rjk, uPrimeOverR, uDoublePrime, laplacian;
    double xk2   = rk[0] * rk[0];
    double yk2   = rk[1] * rk[1];
    double zk2   = rk[2] * rk[2];
    double alpha = m_parameters.at(0);
    double beta  = m_parameters.at(1);
    double a     = m_bosonDiameter;
    // 2 * nabla Phi / Phi:
    std::vector<double> nablaPhi(3, -4 * alpha), gradCorrelation(3, 0), rj(3, 0);
    nablaPhi[0] *= xk2;
    nablaPhi[1] *= yk2;
    nablaPhi[2] *= beta * zk2;
    // First term of the laplacian: nabla^2 Phi / Phi
    laplacian = 4*alpha*alpha*(xk2 + yk2 + beta*beta*zk2);
    laplacian -= 2*alpha*(2 + beta);
    for (int j=0; j<m_system->getNumberOfParticles(); j++) {
        if (j == k) { continue; }
        rjk = computeSingleDistance(particles[k], particles[j]);
        rj = particles[j]->getPosition();
        if (rjk > a) {
            uPrimeOverR = a / (rjk * rjk * (rjk - a));
            gradCorrelation[0] += uPrimeOverR * (rk[0] - rj[0]);
            gradCorrelation[1] += uPrimeOverR * (rk[1] - rj[1]);
            gradCorrelation[2] += uPrimeOverR * (rk[2] - rj[2]);
            // Fourth term of the laplacian
            laplacian += (a*a - 2*a*rjk) / (rjk*rjk * (rjk - a) * (rjk - a)); // Do not optimize this!
            // The line above caused numerical unstabilities when "optimized", I
            // spent roughly a day trying to figure out why the energy kept decreasing
            // with larger alphas. So leave the line as it is :P
            laplacian += uPrimeOverR;
        }
    }
    // Second term of the laplacian
    laplacian += dotProduct(nablaPhi, gradCorrelation);
    // Third term of the laplacian
    laplacian += dotProduct(gradCorrelation, gradCorrelation);
    return laplacian;
}

double Correlated::computeSingleDistance(Particle* p1, Particle* p2) {
    /*
    Compute the distance between particle p1 and p2.
    */
    std::vector<double> pos1 = p1->getPosition();
    std::vector<double> pos2 = p2->getPosition();
    double dist2 = 0;
    for (int i=0; i<3; i++) {
        double diff = pos1[i] - pos2[i];
        dist2 += diff*diff;
    }
    return sqrt(dist2);
}

double Correlated::computeSingleOneBodyPart(Particle* particle) {
    /*
    Computes the one-body WF factor for a single particle
    */
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    std::vector<double> pos = particle->getPosition();
    double sumPos = pos[0]*pos[0] + pos[1]*pos[1];
    sumPos += beta*pos[2]*pos[2];
    return exp(-alpha*sumPos);
}

double Correlated::computeFullOneBodyPart(std::vector<class Particle*> particles) {
    /*
    Computes the one-body WF factor for all particles.
    Each particle has an elliptical Gaussian distribution.
    */
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    double sumPos = 0;
    for(auto particle : particles) {
        std::vector<double> pos = particle->getPosition();
        sumPos += pos[0]*pos[0] + pos[1]*pos[1];
        sumPos += beta*pos[2]*pos[2];
    }
    return exp(-alpha*sumPos);
}

double Correlated::computeSingleInteractingPart(Particle* p1, Particle* p2) {
    /*
    Computes the interaction of two particles with eachother.
    */
    double dist = computeSingleDistance(p1, p2);
    if (dist <= m_bosonDiameter) {
        return 0;
    } else {
        return (1.0 - m_bosonDiameter/dist);
    }
}

double Correlated::evaluateCostFunction() {
    double term1 = m_system->getSampler()->getExpectWfDerivTimesLocalE();
    double term2 = m_system->getSampler()->getExpectWfDerivExpectLocalE();
    double cost = 2*(term1/term2 - 1);
    cost *= term2;
    return cost;
}

std::vector<double> Correlated::computeQuantumForce(class Particle* particle) {
    std::vector<double> vec(3, 0);
    return vec;
}

double Correlated::evaluateDerivative(std::vector<class Particle*> particles) {
    // double deriv;
    // double alpha = m_parameters.at(0);
    // double beta  = m_parameters.at(1);
    // double a     = m_bosonDiameter;
    //
    // for (int k=0; k<particles.size(); k++) {
    //     std::vector<double> rk = particles[k]->getPosition();
    //     double rjk, uPrimeOverR, uDoublePrime, laplacian;
    //     double xk2   = rk[0] * rk[0];
    //     double yk2   = rk[1] * rk[1];
    //     double zk2   = rk[2] * rk[2];
    //     std::vector<double> derivNablaPhi(3, -4), gradCorrelation(3, 0), rj(3, 0);
    //     derivNablaPhi[0] *= xk2;
    //     derivNablaPhi[1] *= yk2;
    //     derivNablaPhi[2] *= beta * zk2;
    //     deriv = 8*alpha*(xk2 + yk2 + beta*beta*zk2);
    //     deriv -= 2*(2 + beta);
    //     for (int j=0; j<m_system->getNumberOfParticles(); j++) {
    //         rjk = computeSingleDistance(particles[k], particles[j]);
    //         rj = particles[j]->getPosition();
    //         if (j == k && rjk > a) {
    //             uPrimeOverR = a / (rjk * rjk * (rjk - a));
    //             gradCorrelation[0] += uPrimeOverR * (rk[0] - rj[0]);
    //             gradCorrelation[1] += uPrimeOverR * (rk[1] - rj[1]);
    //             gradCorrelation[2] += uPrimeOverR * (rk[2] - rj[2]);
    //         }
    //     }
    //     // Derivative of the Second term of the laplacian
    //     deriv += dotProduct(derivNablaPhi, gradCorrelation);
    // }
    // return -0.5 * deriv;
    return - m_system->getSumRiSquared();
}

double Correlated::dotProduct(std::vector<double> v1, std::vector<double> v2) {
    double sum = 0;
    for (int i=0; i<v1.size(); i++) {
        sum += v1[i]*v2[i];
    }
    return sum;
}
