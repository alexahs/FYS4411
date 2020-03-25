#pragma once
#include "initialstate.h"

class RandomUniform : public InitialState {
    /* Random uniform distribution within a box of side lengths, specified
     * by the variable 'characteristicLength'
     */
public:
    RandomUniform(System* system, int numberOfDimensions, int numberOfParticles, double characteristicLength);
    void setupInitialState();
};
