#pragma once
#include "sampler.h"


class WfSampler : public Sampler {
public:
    WfSampler(class System* system);
    void sample(bool acceptedStep);
    void computeAverages();

    // double getExpectWfDerivTimesLocalE()    { return m_expectWfDerivTimesLocalE; }
    // double getExpectWfDerivExpectLocalE()   { return m_expectWfDerivExpectLocalE; }

};
