#pragma once
#include <vector>
#include "neuralquantumstate.h"


class Hamiltonian {
private:
    double m_omega;

public:
    Hamiltonian(double omega);
    double computeLocalEnergy();

    //gradients wrt variational parameters (weights / biases)
    std::vector<double> computeCostGradient();
    double evaluateCost();
}
