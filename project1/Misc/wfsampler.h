#pragma once
#include "sampler.h"


class WfSampler : public Sampler {
public:
    WfSampler(System* system);
    void sample(bool acceptedStep);
    void computeAverages();
    double gradientDescent();

// protected:
//     //for gradient descent
//     double m_wfDerivative = 0;              //<wf^(-1)*dWf/d(params)>
//     double m_expectWfDerivTimesLocalE = 0;  //<wf^(-1)*dWf/d(params)*localEnergy>
//     double m_expectWfDerivExpectLocalE = 0;     //<wf^(-1)*dWf/d(params)>*<localEnergy>
//
//     double m_cumulativeWfDerivative = 0;
//     double m_cumulativeWfDerivTimesLocalE = 0;
};
