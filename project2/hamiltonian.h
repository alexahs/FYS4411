#pragma once
#include <vector>
#include "neuralquantumstate.h"


class Hamiltonian {
private:
    double m_omega;
    int m_nParticles;
    int m_nDims;
    bool m_interaction;


public:
    Hamiltonian(double omega, int nParticles, int nDims, bool interaction);
    double computeLocalEnergy(NeuralQuantumState &nqs);
    double computeInteractionTerm(std::vector<double> nodes);
    //gradients wrt variational parameters (weights / biases)
    std::vector<double> computeCostGradient();
    double evaluateCost();
};
