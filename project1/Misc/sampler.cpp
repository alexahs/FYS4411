#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cassert>
#include "sampler.h"
#include "wfsampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;
using std::setw;
using std::setprecision;

Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
    m_vecEnergySamples.reserve(steps);
}

void Sampler::sample(bool acceptedStep) {
    // Make sure the sampling variable(s) are initialized at the first step.
    if (m_stepNumber == 0) {
        m_cumulativeEnergy = 0;
        m_cumulativeEnergy2 = 0;
        m_acceptedSteps = 0;
        m_acceptRatio = 0;
    }

    if (acceptedStep) { m_acceptedSteps++; }

    double localEnergy = m_system->getHamiltonian()->computeLocalEnergy(m_system->getParticles());
    m_cumulativeEnergy  += localEnergy;
    m_cumulativeEnergy2 += localEnergy*localEnergy;
    m_stepNumber++;
    m_vecEnergySamples.push_back(localEnergy);
}

void Sampler::printOutputToTerminal() {
    int p = m_system->getWaveFunction()->getNumberOfParameters();
    std::vector<double> pa = m_system->getWaveFunction()->getParameters();
    for (int i=0; i<p; i++) { cout << setw(12) << pa.at(i) << "|"; }
    cout << setw(12) << setprecision(6) << m_energy << "|";
    cout << setw(12) << setprecision(6) << m_energy2 << "|";
    cout << setw(12) << setprecision(6) << m_variance << "|";
    cout << setw(12) << setprecision(6) << m_acceptRatio << "\n ";
}

void Sampler::computeAverages() {
    /* Compute the averages of the sampled quantities. You need to think
     * thoroughly through what is written here currently; is this correct?
     */
    int nMetropolisSteps = m_system->getNumberOfMetropolisSteps();
    m_energy = m_cumulativeEnergy / nMetropolisSteps;
    m_energy2 = m_cumulativeEnergy2 / nMetropolisSteps;
    m_variance = m_energy2 - m_energy*m_energy;
    m_acceptRatio = double (m_acceptedSteps) / double (nMetropolisSteps);

    m_wfDerivative = m_cumulativeWfDerivative / nMetropolisSteps;
    m_expectWfDerivTimesLocalE = m_cumulativeWfDerivTimesLocalE / nMetropolisSteps;
    m_expectWfDerivExpectLocalE = m_wfDerivative*m_expectWfDerivTimesLocalE;

}

// void Sampler::initializeParallelVecs(int threadID){
//
//
//     // m_vecSharedEnergySamples.at(threadID)
//
//
//
//
//
//
//
// }
