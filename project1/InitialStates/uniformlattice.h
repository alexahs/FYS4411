#pragma once
#include "initialstate.h"

class UniformLattice : public InitialState {
public:
    UniformLattice(System* system, int numberOfDimensions, int numberOfParticles, double characteristicLength);
    void setupInitialState();

};
