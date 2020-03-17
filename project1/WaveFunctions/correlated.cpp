#include "correlated.h"
#include "Misc/wfsampler.h"
#include <cmath>
#include <cassert>


Correlated::Correlated(System* system, double alpha, double beta, double radius) :
      WaveFunction(system) {
    assert(alpha >= 0);
    assert(beta >= 0);
    assert(system->getNumberOfDimensions() == 3);
    assert(radius > 0);
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_hardShpereRadius = radius;
}

double Correlated::evaluate(std::vector<class Particle*> particles) {

    double wf = 1;

    int N = m_system->getNumberOfParticles();

    for(int i = 0; i < N; i++){
        for(int j = i+1; j < N; j++){
            wf *= computeSingleInteractingPart(particles[i], particles[j]);
        }
    }

    wf *= computeFullOneBodyPart(particles);

    return wf;
}

double Correlated::computeSingleDistance(Particle* p1, Particle* p2){
    double dist = 0;

    std::vector<double> pos1 = p1->getPosition();
    std::vector<double> pos2 = p2->getPosition();

    for(int i = 0; i < m_system->getNumberOfDimensions(); i++){
        double diff = pos1[i] - pos2[i];
        dist += diff*diff;
    }

    return dist;
}


double Correlated::computeSingleOneBodyPart(Particle* particle){
    /*
    Computes the one-body WF factor for a single particle
    */
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    double sumPos = 0;

    std::vector<double> pos = particle->getPosition();
    sumPos += pos[0]*pos[0] + pos[1]*pos[1];
    sumPos += beta*pos[2]*pos[2];

    return exp(-alpha*sumPos);
}


double Correlated::computeFullOneBodyPart(std::vector<class Particle*> particles){
    /*
    Computes the one-body WF factor for all particles
    */
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    double sumPos = 0;

    for(auto particle : particles){
        std::vector<double> pos = particle->getPosition();
        sumPos += pos[0]*pos[0] + pos[1]*pos[1];
        sumPos += beta*pos[2]*pos[2];
    }

    return exp(-alpha*sumPos);
}

double Correlated::computeSingleInteractingPart(Particle* p1, Particle* p2){
    /*
    Computes the interaction of two particles with eachother
    */
    double dist = computeSingleDistance(p1, p2);
    if (dist <= m_hardShpereRadius){
        return 0;
    }
    else{
        return 1.0 - m_hardShpereRadius/dist;
    }
}

double Correlated::computeLaplacian(std::vector<class Particle*> particles){



    return 1;
}



double Correlated::computeDoubleDerivative(std::vector<class Particle*> particles) {


    return 1.0;
}


double Correlated::evaluateCostFunction(){
    return 1.0;
}
