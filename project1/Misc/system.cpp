#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include <cassert>
#include <iostream>
#include <cmath>

using std::cout;
using std::endl;
using std::setw;

bool System::metropolisStep() {
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change it's position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated
     * at this new position with the one at the old position).
     */


    int rndIdx = Random::nextInt(m_numberOfParticles);
    Particle* randomParticle = m_particles.at(rndIdx);
    std::vector<double> proposedSteps = std::vector<double>();

    for (int dim=0; dim<m_numberOfDimensions; dim++){
        double step = m_stepLength*(Random::nextDouble() - 0.5);
        proposedSteps.push_back(step);
        randomParticle->adjustPosition(proposedSteps.at(dim), dim); // Adjust position
    }
    double wfNew = m_waveFunction->evaluate(m_particles); // Evaluate new wave function

    double probabilityRatio = wfNew*wfNew/(wfOld*wfOld);
    // cout << probabilityRatio << endl;

    if (Random::nextDouble() <= probabilityRatio) {
        wfOld = wfNew;
        return true;
    }
    else {
        // If move is rejected, then revert to the old position
        for (int dim=0; dim<m_numberOfDimensions; dim++) {
            randomParticle->adjustPosition(- proposedSteps.at(dim), dim);
        }
        return false;
    }
}

bool System::importanceStep(){


    int rndIdx = Random::nextInt(m_numberOfParticles);
    Particle* randomParticle = m_particles.at(rndIdx);
    std::vector<double> proposedSteps = std::vector<double>();

    //evaluate old quantities
    std::vector<double> posOld = randomParticle->getPosition();
    std::vector<double> qForceOld = m_waveFunction->computeQuantumForce(randomParticle);

    for (int dim=0; dim<m_numberOfDimensions; dim++){
        //calculate new position based on the langevin equation
        double step = qForceOld.at(dim)*m_timeStepDiffusion
                      + Random::nextGaussian(0, 1.0)*m_sqrtTimeStep;
        proposedSteps.push_back(step);
        randomParticle->adjustPosition(proposedSteps.at(dim), dim);
    }

    //evaluate new quantities
    double wfNew = m_waveFunction->evaluate(m_particles);
    std::vector<double> posNew = randomParticle->getPosition();
    std::vector<double> qForceNew = m_waveFunction->computeQuantumForce(randomParticle);





    //compute greens function
    double greensFunctionRatio = 0;
    for (int dim=0; dim<m_numberOfDimensions; dim++){
        double term1 = posOld.at(dim) - posNew.at(dim) - m_timeStepDiffusion*qForceNew.at(dim);
        double term2 = posNew.at(dim) - posOld.at(dim) - m_timeStepDiffusion*qForceOld.at(dim);
        greensFunctionRatio += (term2*term2) - (term1*term1);
    }
    greensFunctionRatio *= m_invFourTimeStepDiffusion;
    greensFunctionRatio = exp(greensFunctionRatio);

    double probabilityRatio = greensFunctionRatio*wfNew*wfNew/(wfOld*wfOld);
    // cout << probabilityRatio << endl;

    //perform test
    if (Random::nextDouble() <= probabilityRatio) {
        wfOld = wfNew;
        return true;
    }
    else {
        for (int dim=0; dim<m_numberOfDimensions; dim++) {
            randomParticle->adjustPosition(- proposedSteps.at(dim), dim);
        }
        return false;
    }
}



void System::runMetropolisSteps(int numberOfMetropolisSteps) {
    m_particles                 = m_initialState->getParticles();
    m_sampler                   = new Sampler(this);
    m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
    m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);
    assert(m_stepLength > 0);

    wfOld = m_waveFunction->evaluate(m_particles);

    //importance sampling
    if (m_importanceSampling) {

        //equilibriate the system
        for (int i=0; i<numberOfMetropolisSteps*m_equilibrationFraction; i++) {
            bool acceptedEquilibriateStep = importanceStep();
        }

        //run with sampling
        for (int i=0; i<numberOfMetropolisSteps; i++) {
            bool acceptedStep = importanceStep();
            m_sampler->sample(acceptedStep);
        }
    }

    //standard metropolis sampling
    else {
        // Equilibriation
        for (int i=0; i<numberOfMetropolisSteps*m_equilibrationFraction; i++) {

            //attempt single step
            bool acceptedEquilibriateStep = metropolisStep();
        }

        for (int i=0; i<numberOfMetropolisSteps; i++) {
            bool acceptedStep = metropolisStep();
            m_sampler->sample(acceptedStep);
        }
    }
    m_sampler->computeAverages();
    m_sampler->printOutputToTerminal();
}

void System::setNumberOfParticles(int numberOfParticles) {
    m_numberOfParticles = numberOfParticles;
}

void System::setNumberOfDimensions(int numberOfDimensions) {
    m_numberOfDimensions = numberOfDimensions;
}

void System::setStepLength(double stepLength) {
    assert(stepLength > 0);
    m_stepLength = stepLength;

}

void System::setEquilibrationFraction(double equilibrationFraction) {
    assert(equilibrationFraction >= 0);
    m_equilibrationFraction = equilibrationFraction;
}

void System::setHamiltonian(Hamiltonian* hamiltonian) {
    m_hamiltonian = hamiltonian;
}

void System::setWaveFunction(WaveFunction* waveFunction) {
    m_waveFunction = waveFunction;
}

void System::setInitialState(InitialState* initialState) {
    m_initialState = initialState;
}

double System::getSumRiSquared() {
    double r2 = 0;
    for (auto particle : m_particles) { // loop over particles
        for (auto xi : particle->getPosition()) { // loop over dimensions
            r2 += xi*xi;
        }
    }
    return r2;
}

void System::setImportanceSampling(bool importanceSampling, double timeStep) {
    if (importanceSampling) {assert(timeStep > 0);}
    m_importanceSampling = importanceSampling;
    m_timeStep = timeStep;
    m_timeStepDiffusion = 0.5*timeStep;
    m_sqrtTimeStep = sqrt(timeStep);
    m_invFourTimeStepDiffusion = 1/(4*m_timeStepDiffusion);
}

void System::setNumericalDoubleDerivative(bool numericalDoubleDerivative, double h) {
    m_numericalDoubleDerivative = numericalDoubleDerivative;
    m_h = h;
}
