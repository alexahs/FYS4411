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

using namespace std;

WfSampler::WfSampler(System* system) :
    Sampler(system) {
    m_system = system;
}


void WfSampler::sample(bool acceptedStep) {
    // Make sure the sampling variable(s) are initialized at the first step.
    if (m_stepNumber == 0) {
        m_cumulativeEnergy = 0;
        m_acceptedSteps = 0;
        m_acceptRatio = 0;
        m_cumulativeWfDerivative = 0;
        m_cumulativeWfDerivTimesLocalE = 0;
    }

    if (acceptedStep) {m_acceptedSteps++;}

    double localEnergy = m_system->getHamiltonian()->
                            computeLocalEnergy(m_system->getParticles());
    double wfDeriv = m_system->getWaveFunction()->
                            evaluateDerivative(m_system->getParticles());

    // cout << localEnergy << endl;

    m_cumulativeEnergy  += localEnergy;
    m_cumulativeEnergy2 += localEnergy*localEnergy;
    m_cumulativeWfDerivative += wfDeriv;
    m_cumulativeWfDerivTimesLocalE += wfDeriv*localEnergy;
    m_stepNumber++;
}

void WfSampler::computeAverages(){
    int nMetropolisSteps = m_system->getNumberOfMetropolisSteps();
    m_energy = m_cumulativeEnergy / nMetropolisSteps;
    m_energy2 = m_cumulativeEnergy2 / nMetropolisSteps;
    m_variance = m_energy2 - m_energy*m_energy;
    m_acceptRatio = double (m_acceptedSteps) / double (nMetropolisSteps);

    m_wfDerivative = m_cumulativeWfDerivative / nMetropolisSteps;
    m_expectWfDerivTimesLocalE = m_cumulativeWfDerivTimesLocalE / nMetropolisSteps;
    // m_expectWfDerivExpectLocalE = m_wfDerivative*m_expectWfDerivTimesLocalE;
    m_expectWfDerivExpectLocalE = m_wfDerivative * m_energy;

    // cout << "deriv        : " << m_wfDerivative << endl;
    // cout << "expect term 1: " << m_expectWfDerivTimesLocalE << endl;
    // cout << "expect term 2: " << m_expectWfDerivExpectLocalE << endl;

}
