#pragma once
#include "hamiltonian.h"
#include <Eigen/Dense>


class Optimizer {
private:
    double m_eta;
    int m_whichOptimizer;

public:
    Optimizer(double eta, int whichOptimizer);
    double getLearningRate() {return m_eta;}

    void optimize(NeuralQuantumState &nqs, Eigen::VectorXd grads, int nInput, int nHidden);
};
