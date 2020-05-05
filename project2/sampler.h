#pragma once
#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "optimizer.h"

class Sampler{
/*
* To maintain a simple and readabla main function, we can let this class be the main class
* responsible for running the metropolis steps and calculate quantities of interest,
* as well as doing the optimization of the weights etc.
* The optimization function itself can ofcourse be outsourced
*/
private:
    int m_nMCcycles;
    int m_nOptimizeIters;
    int m_samplingRule;
    double m_tolerance;

    //loop over mc cycles and sample energies etc
    void runSampling();

public:
    Sampler(int nMCcycles,
            int nOptimizeIters,
            int samplingRule,
            double tolerance,
            Hamiltonian &hamiltonian,
            NeuralQuantumState &nqs,
            Optimizer &optimizer);

    //loop over gradient descent steps, calling runSampling each iteration
    void runOptimization();

    Hamiltonian m_hamiltonian;
    NeuralQuantumState m_nqs;
    Optimizer &m_optimizer;
};
