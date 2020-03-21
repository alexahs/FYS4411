#pragma once
#include "sampler.h"

class WfSampler : public Sampler {
public:
    WfSampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void sample(bool acceptedStep);
    void computeAverages();
};
