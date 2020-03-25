#include <iostream>
#include <iomanip>
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
#include "Misc/writefile.h"

using namespace std;

Sampler::Sampler(System* system) {
    /* Contructor */
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setOneBodyDensity(double min, double max, int numberOfBins) {
    m_numberOfBins = numberOfBins;
    m_min = min;
    m_max = max;
    m_binWidth = (max - min) / numberOfBins;
    m_bins = (double**) calloc(m_system->getNumberOfDimensions(), sizeof(double*));
    for (int i=0; i<m_system->getNumberOfDimensions(); i++) {
        m_bins[i] = (double*) calloc(m_numberOfBins, sizeof(double));
    }
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

    /* Estimate the One-body density integral with a sum */
    if (m_numberOfBins) {
        std::vector<double> pos; int idx; double xi;
        for (auto particle : m_system->getParticles()) {
            pos = particle->getPosition();
            for (int d=0; d<m_system->getNumberOfDimensions(); d++) {
                xi = pos[d];
                if (xi < m_min)         { idx = 0; }
                else if (xi > m_max)    { idx = m_numberOfBins - 1; }
                else                    { idx = (int) floor((xi - m_min) / m_binWidth); }
                m_bins[d][idx] += 1.0;
            }
        }
    }
}

void Sampler::computeAverages() {
    int nMetropolisSteps = m_system->getNumberOfMetropolisSteps();
    m_energy = m_cumulativeEnergy / nMetropolisSteps;
    m_energy2 = m_cumulativeEnergy2 / nMetropolisSteps;
    m_variance = m_energy2 - m_energy*m_energy;
    m_acceptRatio = double (m_acceptedSteps) / double (nMetropolisSteps);
    m_wfDerivative = m_cumulativeWfDerivative / nMetropolisSteps;
    m_expectWfDerivTimesLocalE = m_cumulativeWfDerivTimesLocalE / nMetropolisSteps;
    m_expectWfDerivExpectLocalE = m_wfDerivative*m_expectWfDerivTimesLocalE;
}

void Sampler::finishOneBodyDensity(string filename) {
    if (m_numberOfBins) {
        writeOneBodyDensity(m_bins, m_numberOfBins, filename);
        free (m_bins);
    }
}
