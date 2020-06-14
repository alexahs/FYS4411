#pragma once
#include <vector>
#include "neuralquantumstate.h"


class Hamiltonian {
private:
    double m_omega;
    bool m_interaction;
    int m_nParticles;
    int m_nDims;
    int m_nInput;


public:
    Hamiltonian(double omega, bool interaction, NeuralQuantumState &nqs);
    double computeLocalEnergy(NeuralQuantumState &nqs);
};
