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

    for(int j = 0; j < m_nHidden; j++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += m_inputLayer[i]*m_weights[i][j];
        }
        term1 /= m_sigma2;
        term1 += m_inputBias[j];
        psi2 *= 1 + exp(term1);
    }

    double wf = exp(-1/(2*m_sigma2)*psi1)*psi2;
    return wf;
}

double NeuralQuantumState::computeNabla2(){
    double gradient = 1;
    return gradient;
}

std::vector<double> NeuralQuantumState::computeQuantumForce(){
    std::vector<double> qForce = {1, 2, 3};
    return qForce;
}
