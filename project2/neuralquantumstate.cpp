#include "neuralquantumstate.h"
#include "Math/random.h"
#include "netparams.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using std::cout;
using std::endl;


NeuralQuantumState::NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma, long seed){
    //
    m_nVisible = (int) nParticles*nDims;
    m_nInput = (int) nParticles*nDims;
    m_nHidden = nHidden;
    m_nParticles = nParticles;
    m_nDims = nDims;
    m_sigma = sigma;
    m_sigma2 = sigma*sigma;
    m_sigma4 = m_sigma2*m_sigma2;
    net = NetParams(m_nInput, m_nHidden);
    initialize();
}



void NeuralQuantumState::initialize(){
    //randomly initialize weights or something

    double sigma_init = 0.01;
    for(int i = 0; i < m_nVisible; i++){
        net.inputBias(i) = Random::nextGaussian(0, sigma_init);
        net.inputLayer(i) = Random::nextDouble() - 0.5;
        for(int j = 0; j < m_nHidden; j++){
            net.weights(i, j) = Random::nextGaussian(0, sigma_init);

        }
    }

    for(int i = 0; i < m_nHidden; i++){
        net.hiddenBias(i) = Random::nextGaussian(0, sigma_init);
    }

}


void NeuralQuantumState::adjustPosition(int node, double change){
    net.inputLayer(node) += change;
}


double NeuralQuantumState::evaluate(){
    //evaluate the wavefunction
    //could probably use some matrix magic to speed up computations here..
    double psi1 = 0;
    double psi2 = 1;
    for(int i = 0; i < m_nVisible; i++){
        psi1 +=  (net.inputLayer(i)-net.inputBias(i))*(net.inputLayer(i)-net.inputBias(i));
    }
    psi1 = exp(-1/(2*m_sigma2)*psi1);


    for(int j = 0; j < m_nHidden; j++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += net.inputLayer(i)*net.weights(i,j);
        }
        term1 /= m_sigma2;
        term1 += net.hiddenBias(j);
        psi2 *= 1 + exp(term1);
    }

    return psi1*psi2;
}

Eigen::VectorXd NeuralQuantumState::computeQfactor(){
    //computes the exponential factor used many times throughout the program
    // exp(b_n - sum(x_i*w_ij)), as seen in equation (102) in notes
    Eigen::VectorXd Qfactor(m_nHidden);
    for(int n = 0; n < m_nHidden; n++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += net.inputLayer(i)*net.weights(i,n);
        }

        Qfactor(n) = exp(net.hiddenBias(n) + term1/m_sigma2);
    }

    return Qfactor;
}


Eigen::VectorXd NeuralQuantumState::computeFirstAndSecondDerivatives(int nodeNumber){
    // returns vector of d/dx_m [ln(psi)] and d^2/dx_m^2 [ln(psi)]. m = nodeNumber
    // equations (115) and (116) in lecture notes

    int m = nodeNumber;

    double dx = 0;
    double ddx = 0;
    Eigen::VectorXd Q = computeQfactor();

    for(int n = 0; n < m_nHidden; n++){
        dx += net.weights(m,n)/(Q(n)+1);
        ddx += net.weights(m,n)*Q(n)/((Q(n)+1)*(Q(n)+1));
    }

    dx /= m_sigma2;
    ddx /= m_sigma4;

    dx += -1/m_sigma2*(net.inputLayer(m) - net.inputBias(m));
    ddx += -1/m_sigma2;

    Eigen::VectorXd derivatives(2);
    derivatives(0) = dx;
    derivatives(1) = ddx;

    return derivatives;
}


double NeuralQuantumState::computeDistance(int p, int q){
    //compute the distance between two particles p and q
    double distance2 = 0;
    int pIdx = p*m_nDims;
    int qIdx = q*m_nDims;
    for(int d = 0; d < m_nDims; d++){
        double distance = 0;
        distance += net.inputLayer(pIdx + d) - net.inputLayer(qIdx + d);
        distance2 += distance*distance;
    }

    return sqrt(distance2);

}


void NeuralQuantumState::computeCostGradient(){
    //gradients wrt variational parameters (weights / biases)
    //equations 101, 102 and 103 in lecture notes
    Eigen::VectorXd Q = computeQfactor();

    //derivative wrt. input bias
    // for(int m = 0; m < m_nVisible; m++){
        //     net.dInputBias[m] = (net.inputLayer[m] -net.inputBias[m])/m_sigma2;
        // }
    net.dInputBias = net.inputLayer - net.inputBias;
    net.dInputBias /= m_sigma2;


    //derivative wrt. hidden bias
    for(int n = 0; n < m_nHidden; n++){
        net.dHiddenBias(n) = 1/(Q(n) + 1);
    }

    //derivative wrt weights
    for(int m = 0; m < m_nVisible; m++){
        for(int n = 0; n < m_nHidden; n++){
            net.dWeights(m,n) = net.inputLayer(m)/(Q(n) + 1)/m_sigma2;
        }
    }

}




Eigen::VectorXd NeuralQuantumState::computeQuantumForce(){
    Eigen::VectorXd qForce(3);
    return qForce;
}
