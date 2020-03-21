#include "correlated.h"
#include "Misc/wfsampler.h"
#include <cmath>
#include <cassert>

#include <iostream>
using namespace std;

Correlated::Correlated(System* system, double alpha, double beta) :
      WaveFunction(system) {
    assert(alpha >= 0);
    assert(beta >= 0);
    assert(system->getNumberOfDimensions() == 3);
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
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
    double term1=0, term2=0, term3=0, term4=0;
    std::vector<double> rk = particles[k]->getPosition();
    double xk2 = rk[0] * rk[0];
    double yk2 = rk[1] * rk[1];
    double zk2 = rk[2] * rk[2];
    double rjk, uPrimeOverR, uDoublePrime;
    double alpha = m_parameters.at(0);
    double beta = m_parameters.at(1);
    double a = m_bosonDiameter;
    // 2 * nabla Phi / Phi:
    std::vector<double> nablaPhi(3, -4 * alpha);
    nablaPhi[0] *= xk2;
    nablaPhi[1] *= yk2;
    nablaPhi[2] *= beta * zk2;
    // First term of the laplacian: nabla^2 Phi / Phi
    double laplacian = 2*alpha*(2*alpha*(xk2 + yk2 + beta*beta*zk2) - beta - 2);
    int num = m_system->getNumberOfParticles();
    std::vector<double> gradCorrelation(3, 0), rj(3, 0);

    for (int j=0; j<num; j++) {
        if (j == k) { continue; }
        rjk = computeSingleDistance(particles[k], particles[j]);
        rj = particles[j]->getPosition();
        if (rjk > a) {
            uPrimeOverR = a / (rjk * rjk * (rjk - a));
            gradCorrelation[0] += uPrimeOverR * (rk[0] - rj[0]);
            gradCorrelation[1] += uPrimeOverR * (rk[1] - rj[1]);
            gradCorrelation[2] += uPrimeOverR * (rk[2] - rj[2]);
            // Fourth term of the laplacian
            uDoublePrime = uPrimeOverR * (a - 2*rjk) / (a - rjk);
            laplacian += uDoublePrime + 2*uPrimeOverR;
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
    return 0;
}

std::vector<double> Correlated::computeQuantumForce(class Particle* particle) {
    std::vector<double> vec(3, 0);
    return vec;
}

double Correlated::evaluateDerivative(std::vector<class Particle*> particles) {
    return 0;
}

double Correlated::dotProduct(std::vector<double> v1, std::vector<double> v2) {
    double sum = 0;
    for (int i=0; i<v1.size(); i++) {
        sum += v1[i]*v2[i];
    }
    return sum;
}
