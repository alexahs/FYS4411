#pragma once
#include <vector>
#include <netparams.h>



class NeuralQuantumState {
private:

    void initialize();
    int m_nHidden = 0;
    int m_nVisible = 0;
    int m_nInput = 0;
    int m_nParticles = 0;
    int m_nDims = 0;
    double m_sigma_init;
    double m_sigma;
    double m_sigma2;
    double m_sigma4;
    bool m_isGibbsSampling = false;

public:
    NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma, long seed, double sigma_init, int samplingRule);

    double evaluate(); //evaluate the wavefunction
    double computeDistance(int p, int q); //distance between two particles p and q
    Eigen::VectorXd computeFirstAndSecondDerivatives(int nodeNumber);
    Eigen::VectorXd computeQfactor();
    Eigen::VectorXd computeCostGradient();
    void adjustPosition(int node, double change); //adjust position of a single particle q

    NetParams net;


    int getNumberOfParticles() {return m_nParticles;}
    int getNumberOfDims()      {return m_nDims;}
    int getNumberOfInputs()    {return m_nVisible;}
    int getNumberOfHidden()    {return m_nHidden;}
    double getSigma2()         {return m_sigma2;}
    double getSigma()         {return m_sigma;}


};
