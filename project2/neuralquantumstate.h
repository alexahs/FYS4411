#pragma once
#include <vector>

struct s_dPsi {
    std::vector<double> dInputBias;
    std::vector<double> dHiddenBias;
    std::vector<std::vector<double>> dWeights;
};

class NeuralQuantumState {
private:

    void initializeWeights();
    void initializePositions();
    int m_nHidden;
    int m_nVisible;
    int m_nParticles;
    int m_nDims;
    double m_sigma;
    double m_sigma2;
    double m_sigma4;
    s_dPsi m_grads;

public:
    NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma, long seed);

    double evaluate(); //evaluate the wavefunction
    double computeDistance(int p, int q); //distance between two particles p and q
    std::vector<double> computeFirstAndSecondDerivatives(int nodeNumber);
    std::vector<double> computeQuantumForce();
    std::vector<double> computeQfactor();
    s_dPsi computeCostGradient();

    std::vector<double> m_inputLayer;
    std::vector<double> m_hiddenLayer;
    std::vector<double> m_inputBias;
    std::vector<double> m_hiddenBias;
    std::vector<std::vector<double>> m_weights; //shape [m_nVisible, m_nHidden]


};
