#pragma once
#include <vector>


class NeuralQuantumState {
private:

    void initializeWeights();
    void initializePositions();
    int m_nHidden;
    int m_nVisible;
    double m_sigma;
    double m_sigma2;
    double m_sigma4;

public:
    NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma, long seed);

    double evaluate(); //evaluate the wavefunction
    std::vector<double> computeFirstAndSecondDerivatives(int nodeNumber);
    std::vector<double> computeQuantumForce();

    std::vector<double> m_inputLayer;
    std::vector<double> m_hiddenLayer;
    std::vector<double> m_inputBias;
    std::vector<double> m_hiddenBias;
    std::vector<std::vector<double>> m_weights; //shape [m_nVisible, m_nHidden]


};
