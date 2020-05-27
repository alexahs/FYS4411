#pragma once
#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "optimizer.h"
#include "netparams.h"
#include <fstream>


class Sampler{
/*
* To maintain a simple and readabla main function, we can let this class be the main class
* responsible for running the metropolis steps and calculate quantities of interest,
* as well as doing the optimization of the weights etc.
* The optimization function itself can ofcourse be outsourced
*/
private:
    int m_nMCcycles;
    int m_samplingRule;
    double m_tolerance;
    int m_nOptimizeIters;
    double m_stepLength;
    double m_timeStepDiffusion;
    double m_sqrtTimeStep;

    int m_acceptedSteps = 0;
    double m_acceptRatio = 0;
    double m_energy = 0;
    double m_energy2 = 0;
    double m_variance = 0;
    double m_wfOld = 0;

    int m_nDims;
    int m_nParticles;
    int m_nHidden;
    int m_nInput;

    Eigen::VectorXd m_dPsi;
    Eigen::VectorXd m_dPsiTimesE;
    Eigen::VectorXd m_costGradient;

    Eigen::VectorXd m_energyVals;
    Eigen::VectorXd m_energy2Vals;
    Eigen::VectorXd m_varianceVals;
    Eigen::VectorXd m_acceptRatioVals;





    //loop over mc cycles and sample energies etc
    void runSampling();
    bool sample(int particleNumber);
    bool metropolisStep(int particleNumber);
    bool importanceStep(int particleNumber);

    void printInitalSystemInfo();
    void printInfo(int step);
    void printFinalInfo();
    void writeFileCumulativeResults();

public:
    Sampler(int nMCcycles,
            int samplingRule,
            double tolerance,
            int nOptimizeIters,
            double stepLength,
            double timeStep,
            Hamiltonian &hamiltonian,
            NeuralQuantumState &nqs,
            Optimizer &optimizer);

    //loop over gradient descent steps, calling runSampling each iteration
    void runOptimization();
    void runDataCollection(int nMCcycles);
    Hamiltonian m_hamiltonian;
    NeuralQuantumState m_nqs;
    Optimizer m_optimizer;
};
