#include "correlated.h"
#include "Misc/wfsampler.h"
#include <cmath>
#include <cassert>

#include <iostream>


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
    std::vector<double> pos_k = particles[k]->getPosition();
    double xk2 = pos_k[0] * pos_k[0];
    double yk2 = pos_k[1] * pos_k[1];
    double zk2 = pos_k[2] * pos_k[2];
    double nabla2Phi, uPrime, uDoublePrime;
    double alpha = m_parameters.at(0);
    double beta = m_parameters.at(1);
    double minus2Alpha = -2*alpha;
    std::vector<double> nablaPhi(3, 0);
    nablaPhi[0] = 2*minus2Alpha * xk2;
    nablaPhi[1] = 2*minus2Alpha * yk2;
    nablaPhi[2] = 2*beta * minus2Alpha * zk2;
    nabla2Phi = 2 * alpha * (2 * alpha * (xk2 + yk2 + beta * beta * zk2) - beta - 2);
    // First term of the laplacian
    double laplacian = nabla2Phi;
    double rjk, rik, fac_j, fac_i;
    int num = m_system->getNumberOfParticles();
    std::vector<double> sumVec(3, 0);
    std::vector<double> temp_j(3, 0);
    std::vector<double> temp_i(3, 0);
    std::vector<double> pos_i(3, 0);
    std::vector<double> pos_j(3, 0);

    for (int j=0; j<num; j++) {
        if (j != k) {
            pos_j = particles[j]->getPosition();
            // r_k - r_j
            temp_j[0] = pos_k[0] - pos_j[0];
            temp_j[1] = pos_k[1] - pos_j[1];
            temp_j[2] = pos_k[2] - pos_j[2];
            // |r_k -  r_j|
            rjk = sqrt(temp_j[0]*temp_j[0] + temp_j[1]*temp_j[1] + temp_j[2]*temp_j[2]);
            if (rjk > m_bosonDiameter) {
                fac_j = 1 / (rjk - m_bosonDiameter);
            } else {
                fac_j = 0;
            }
            // fac_j = u'(rjk) / rjk
            sumVec[0] += fac_j * temp_j[0];
            sumVec[1] += fac_j * temp_j[1];
            sumVec[2] += fac_j * temp_j[2];
            // sumVec is the entire sum in the 2nd term of the laplacian,
            // this is going to be dotted with 2* nablaPhi / phi

            for (int i=0; i<num; i++) {
                double dot=0;
                if (i != k) {
                    pos_i = particles[i]->getPosition();
                    temp_i[0] = pos_k[0] - pos_i[0];
                    temp_i[1] = pos_k[1] - pos_i[1];
                    temp_i[2] = pos_k[2] - pos_i[2];
                    dot += temp_j[0]*temp_i[0];
                    dot += temp_j[1]*temp_i[1];
                    dot += temp_j[2]*temp_i[2];
                    rik = sqrt(temp_i[0]*temp_i[0] + temp_i[1]*temp_i[1] + temp_i[2]*temp_i[2]);
                    fac_i = 1 / (rik - m_bosonDiameter);
                    // Term 3 in the equation for the laplacian
                    laplacian += dot*fac_i*fac_j;
                }
            }

            // u''(rjk) = a(a - 2rjk) / ( rjk^2 (a - rjk)^2 )
            uDoublePrime = fac_j * fac_j * m_bosonDiameter;
            uDoublePrime *= (m_bosonDiameter - 2*rjk);
            uDoublePrime /= rjk*rjk;
            // Term 4 in the equation for the laplacian
            laplacian += uDoublePrime + 2 * fac_j;
        }
    }
    // Second term of the laplacian (dot product)
    laplacian += nablaPhi[0]*sumVec[0] + nablaPhi[1]*sumVec[1] + nablaPhi[2]*sumVec[2];

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
