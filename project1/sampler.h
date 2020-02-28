#pragma once

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void sample(bool acceptedStep);
    void printOutputToTerminal();
    void computeAverages();
    double getEnergy()          { return m_energy; }
    double getEnergy2()         { return m_energy2; }
    double getVariance()        { return m_variance; }

private:
    int     m_numberOfMetropolisSteps = 0;
    int     m_stepNumber = 0;
    double  m_energy = 0;
    double  m_energy2 = 0;
    double  m_variance = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy2 = 0;
    class System* m_system = nullptr;
};
