#include "neuralquantumstate.h"
#include "Math/random.h"
#include <iostream>
#include <cmath>

using std::cout;
using std::endl;


NeuralQuantumState::NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma, long seed){
    //
    m_nVisible = (int) nParticles*nDims;
    m_nHidden = nHidden;
    m_sigma = sigma;
    m_sigma2 = sigma*sigma;
    m_sigma4 = m_sigma2*m_sigma2;
    m_inputLayer.reserve(m_nVisible);
    m_hiddenLayer.reserve(m_nHidden);
    m_inputBias.reserve(m_nVisible);
    m_hiddenBias.reserve(m_nHidden);
    m_weights.reserve(m_nVisible);
    for(int i = 0; i < m_nVisible; i++){
        m_weights[i].reserve(m_nHidden);
    }
    initializeWeights();
    initializePositions();
}



void NeuralQuantumState::initializeWeights(){
    //randomly initialize weights or something
    double sigma_init = 0.01;
    for(int i = 0; i < m_nVisible; i++){
        m_inputBias[i] = Random::nextGaussian(0, sigma_init);
        for(int j = 0; j < m_nHidden; j++){
            m_weights[i][j] = Random::nextGaussian(0, sigma_init);

        }
    }

    for(int i = 0; i < m_nHidden; i++){
        m_hiddenBias[i] = Random::nextGaussian(0, sigma_init);
    }

}

void NeuralQuantumState::initializePositions(){
    //initialize positions
    for(int i = 0; i < m_nVisible; i++){
        m_inputLayer[i] = Random::nextDouble() - 0.5;
        cout << m_inputLayer[i] << endl;
    }

}


double NeuralQuantumState::evaluate(){
    //evaluate the wavefunction
    //could probably use some matrix magic to speed up computations here..

    double psi1 = 0;
    double psi2 = 1;
    for(int i = 0; i < m_nVisible; i++){
        psi1 += (m_inputLayer[i]-m_inputBias[i])*(m_inputLayer[i]-m_inputBias[i]);
    }
    psi1 = exp(-1/(2*m_sigma2)*psi1);

    for(int j = 0; j < m_nHidden; j++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += m_inputLayer[i]*m_weights[i][j];
        }
        term1 /= m_sigma2;
        term1 += m_inputBias[j];
        psi2 *= 1 + exp(term1);
    }

    return psi1*psi2;
}


std::vector<double> NeuralQuantumState::computeFirstAndSecondDerivatives(int nodeNumber){
    // returns vector of d/dx_m [ln(psi)] and d^2/dx_m^2 [ln(psi)]. m = nodeNumber

    int m = nodeNumber;

    double dx = 0;
    double ddx = 0;

    for(int n = 0; n < m_nHidden; n++){
        double Q = 0;
        double term1 = 0;
        for(int i = 0; n < m_nVisible; i++){
            term1 += m_inputLayer[i]*m_weights[i][n];
        }

        Q = exp(m_hiddenBias[n] + term1/m_sigma2);

        dx += m_weights[m][n]/(Q+1);
        ddx += m_weights[m][n]*Q/((Q+1)*(Q+1));
    }

    dx /= m_sigma2;
    ddx /= m_sigma4;

    dx += -1/m_sigma2*(m_inputLayer[m] - m_inputBias[m]);
    ddx += -1/m_sigma2;

    std::vector<double> derivatives = {dx, ddx};

    return derivatives;
}


std::vector<double> NeuralQuantumState::computeQuantumForce(){
    std::vector<double> qForce = {1, 2, 3};
    return qForce;
}
