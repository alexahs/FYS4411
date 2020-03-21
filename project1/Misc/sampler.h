#pragma once
#include <iomanip>
#include <vector>

class Sampler {
public:
    Sampler(class System* system);
    virtual void setNumberOfMetropolisSteps(int steps);
    virtual void sample(bool acceptedStep);
    virtual void computeAverages();
    // void printOutputToTerminal();
    double getEnergy()                      { return m_energy; }
    double getEnergy2()                     { return m_energy2; }
    double getVariance()                    { return m_variance; }
    double getAcceptRatio()                 { return m_acceptRatio; }
    double getExpectWfDerivTimesLocalE()    { return m_expectWfDerivTimesLocalE; }
    double getExpectWfDerivExpectLocalE()   { return m_expectWfDerivExpectLocalE; }
    std::vector <double> getEnergySamples() { return m_vecEnergySamples; }

protected:
    int     m_numberOfMetropolisSteps = 0;
    int     m_stepNumber = 0;
    int     m_acceptedSteps = 0;
    double  m_acceptRatio = 0;
    double  m_energy = 0;
    double  m_energy2 = 0;
    double  m_variance = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy2 = 0;
    class System* m_system = nullptr;
    bool m_saveEnergySamples = false;
    double m_wfDerivative = 0;                  //  <wf^(-1)*dWf/d(params)>
    double m_expectWfDerivTimesLocalE = 0;      //  <wf^(-1)*dWf/d(params)*localEnergy>
    double m_expectWfDerivExpectLocalE = 0;     //  <wf^(-1)*dWf/d(params)>*<localEnergy>
    double m_cumulativeWfDerivative = 0;
    double m_cumulativeWfDerivTimesLocalE = 0;
    std::vector <double> m_vecEnergySamples;
};
