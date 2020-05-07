#pragma once
#include "hamiltonian.h"


class Optimizer {
private:
    double m_eta;
    int m_whichOptimizer;

public:
    Optimizer(double eta, int whichOptimizer);


    void optimize(NeuralQuantumState &nqs);
};
