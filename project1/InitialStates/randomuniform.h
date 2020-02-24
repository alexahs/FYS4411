#ifndef RANDUMUNIFORM_H
#define RANDUMUNIFORM_H

#include "initialstate.h"

class RandomUniform : public InitialState
{
public:
    RandomUniform(System* system, int numberOfDimensions, int numberOfParticles);
    void setupInitialState();
};

#endif //RANDUMUNIFORM_H
