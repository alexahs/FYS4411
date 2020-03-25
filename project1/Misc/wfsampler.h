#pragma once
#include "sampler.h"

class WfSampler : public Sampler {
    /* This class is derived from Sampler, and its purpose is to sample
     * some different statistics than the other sampler class. It is meant
     * to be used during Gradient Descent
     */
public:
    WfSampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void sample(bool acceptedStep);
    void computeAverages();
};
