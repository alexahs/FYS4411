#pragma once
#include "initialstate.h"

class UniformLattice : public InitialState {
    /* Uniform initialization into a lattice (grid), this can be convenient
     * for very large systems, because particles would not be initialized
     * on top of each other.
     */
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
