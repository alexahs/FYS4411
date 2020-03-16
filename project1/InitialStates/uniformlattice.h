#pragma once
#include "initialstate.h"

class UniformLattice : public InitialState {
public:
    UniformLattice(System* system,
                   int numberOfDimensions,
                   int numberOfParticles,
                   double characteristicLength,
                   double hardSphereRadius);
    void setupInitialState();

private:
    double m_hardShpereRadius = 1;

};
