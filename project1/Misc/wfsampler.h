#pragma once
#include "sampler.h"


class WfSampler : public Sampler {
public:
    WfSampler(System* system);
    void sample(bool acceptedStep);
    void computeAverages();

    
};
