#pragma once
#include <vector>
#include <netparams.h>

struct s_dPsi {
    std::vector<double> dInputBias;
    std::vector<double> dHiddenBias;
    std::vector<std::vector<double>> dWeights;
};


class NeuralQuantumState {
private:

    void initialize();
    int m_nHidden = 0;
    int m_nVisible = 0;
    int m_nInput = 0;
    int m_nParticles = 0;
    int m_nDims = 0;
    double m_sigma;
    double m_sigma2;
    double m_sigma4;

public:
    NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma, long seed);

    double evaluate(); //evaluate the wavefunction
    double computeDistance(int p, int q); //distance between two particles p and q
    std::vector<double> computeFirstAndSecondDerivatives(int nodeNumber);
    std::vector<double> computeQuantumForce();
    std::vector<double> computeQfactor();
    s_dPsi computeCostGradient();
    void adjustPosition(int node, double change); //adjust position of a single particle q


    NetParams m_params;


    std::vector<double> m_inputLayer;
    std::vector<double> m_hiddenLayer;
    std::vector<double> m_inputBias;
    std::vector<double> m_hiddenBias;
    std::vector<std::vector<double>> m_weights; //shape [m_nVisible, m_nHidden]
    s_dPsi m_grads;

    int getNumberOfParticles() {return m_nParticles;}
    int getNumberOfDims()      {return m_nDims;}
    int getNumberOfInputs()    {return m_nVisible;}
    int getNumberOfHidden()    {return m_nHidden;}


};
