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
    m_nParticles = nParticles;
    m_nDims = nDims;
    m_sigma = sigma;
    m_sigma2 = sigma*sigma;
    m_sigma4 = m_sigma2*m_sigma2;
    m_inputLayer.resize(m_nVisible);
    m_hiddenLayer.resize(m_nHidden);
    m_inputBias.resize(m_nVisible);
    m_hiddenBias.resize(m_nHidden);
    m_weights.resize(m_nVisible);
    for(int i = 0; i < m_nVisible; i++){
        m_weights[i].resize(m_nHidden);
    }

    m_grads.dInputBias.resize(m_nVisible);
    m_grads.dHiddenBias.resize(nHidden);
    m_grads.dWeights.resize(m_nVisible);
    for(int i = 0; i < m_nVisible; i++){
        m_grads.dWeights[i].resize(nHidden);
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

std::vector<double> NeuralQuantumState::computeQfactor(){
    //computes the exponential factor used many times throughout the program
    // exp(b_n - sum(x_i*w_ij)), as seen in equation (102) in notes
    std::vector<double> Qfactor(m_nHidden);
    for(int n = 0; n < m_nHidden; n++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += m_inputLayer[i]*m_weights[i][n];
        }

        Qfactor[n] = exp(m_hiddenBias[n] + term1/m_sigma2);
    }

    return Qfactor;
}


std::vector<double> NeuralQuantumState::computeFirstAndSecondDerivatives(int nodeNumber){
    // returns vector of d/dx_m [ln(psi)] and d^2/dx_m^2 [ln(psi)]. m = nodeNumber
    // equations (115) and (116) in lecture notes

    int m = nodeNumber;

    double dx = 0;
    double ddx = 0;

    std::vector<double> Q = computeQfactor();

    for(int n = 0; n < m_nHidden; n++){
        dx += m_weights[m][n]/(Q[n]+1);
        ddx += m_weights[m][n]*Q[n]/((Q[n]+1)*(Q[n]+1));
    }

    dx /= m_sigma2;
    ddx /= m_sigma4;

    dx += -1/m_sigma2*(m_inputLayer[m] - m_inputBias[m]);
    ddx += -1/m_sigma2;

    std::vector<double> derivatives = {dx, ddx};

    return derivatives;
}


double NeuralQuantumState::computeDistance(int p, int q){
    //compute the distance between two particles p and q
    double distance2 = 0;
    int pIdx = p*m_nDims;
    int qIdx = q*m_nDims;
    for(int d = 0; d < m_nDims; d++){
        double distance = 0;
        distance += m_inputLayer[pIdx + d] - m_inputLayer[qIdx + d];
        distance2 += distance*distance;
    }

    return sqrt(distance2);

}


s_dPsi NeuralQuantumState::computeCostGradient(){
    //gradients wrt variational parameters (weights / biases)
    //equations 101, 102 and 103 in lecture notes
    std::vector<double> Q = computeQfactor();

    //derivative wrt. input bias
    for(int m = 0; m < m_nVisible; m++){
        m_grads.dInputBias[m] = (m_inputLayer[m] -m_inputBias[m])/m_sigma2;
    }

    //derivative wrt. hidden bias
    for(int n = 0; n < m_nHidden; n++){
        m_grads.dHiddenBias[n] = 1/(Q[n] + 1);
    }

    //derivative wrt weights
    for(int m = 0; m < m_nVisible; m++){
        for(int n = 0; n < m_nHidden; n++){
            m_grads.dWeights[m][n] = m_inputLayer[m]/(Q[n] + 1)/m_sigma2;
        }
    }

    return m_grads;

}




std::vector<double> NeuralQuantumState::computeQuantumForce(){
    std::vector<double> qForce = {1, 2, 3};
    return qForce;
}
