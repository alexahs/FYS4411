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
    // Random::setSeed(100);
    double sigma_init = 0.001;
    for(int i = 0; i < m_nVisible; i++){
        net.inputBias(i) = Random::nextGaussian(0, sigma_init);
        net.inputLayer(i) = Random::nextDouble() - 0.5;
        // std::cout << net.inputBias(i) << std::endl;
        for(int j = 0; j < m_nHidden; j++){
            net.weights(i, j) = Random::nextGaussian(0, sigma_init);

        }
    }
    // exit(1);

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
    psi1 = exp(-psi1/(2*m_sigma2));



    //////
    Eigen::VectorXd Q = computeQfactor();
    for(int j = 0; j<m_nHidden; j++){
        psi2 *= (1 + exp(Q(j)));
    }
    //////

    /*
    for(int j = 0; j < m_nHidden; j++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += net.inputLayer(i)*net.weights(i,j);
        }
        term1 /= m_sigma2;
        term1 += net.hiddenBias(j);
        psi2 *= 1 + exp(term1);
    }
    */

    return psi1*psi2;
}

Eigen::VectorXd NeuralQuantumState::computeQfactor(){
    //computes the exponential factor used many times throughout the program
    // b_j - sum(x_i*w_ij), as seen in equation (102) in notes

    /*
    Eigen::VectorXd Qfactor(m_nHidden);
    for(int n = 0; n < m_nHidden; n++){
        double term1 = 0;
        for(int i = 0; i < m_nVisible; i++){
            term1 += net.inputLayer(i)*net.weights(i,n);
        }

        Qfactor(n) = exp(net.hiddenBias(n) + term1/m_sigma2);
    }
    */

    Eigen::VectorXd Qfactor = net.hiddenBias + (((1.0/m_sigma2)*net.inputLayer).transpose()*net.weights).transpose();

    return Qfactor;
}


Eigen::VectorXd NeuralQuantumState::computeFirstAndSecondDerivatives(int nodeNumber){
    // returns vector of d/dx_m [ln(psi)] and d^2/dx_m^2 [ln(psi)]. m = nodeNumber
    // equations (115) and (116) in lecture notes

    int m = nodeNumber;

    double sum1 = 0;
    double sum2 = 0;
    Eigen::VectorXd Q = computeQfactor();

    for(int n = 0; n < m_nHidden; n++){
        double exp_Q = exp(Q(n));
        double exp_neg_Q = exp(-Q(n));

        sum1 += net.weights(m, n)/(exp_neg_Q + 1);
        sum2 += net.weights(m, n)*net.weights(m, n)*exp_Q/((exp_Q + 1)*(exp_Q + 1));

    }

    double dx = (-(net.inputLayer(m) - net.inputBias(m)) + sum1)/m_sigma2;
    double ddx = -1/m_sigma2 + sum2/m_sigma4;

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


Eigen::VectorXd NeuralQuantumState::computeCostGradient(){
    //gradients wrt variational parameters (weights / biases)
    //equations 101, 102 and 103 in lecture notes
    Eigen::VectorXd grads(m_nInput + m_nHidden + m_nInput*m_nHidden);


    Eigen::VectorXd Q = computeQfactor();
    Eigen::VectorXd exp_neg_Q(m_nHidden);
    for(int i = 0; i < m_nHidden; i++){
        exp_neg_Q(i) = exp(-Q(i));
    }


    //derivative wrt. input bias
    for(int i = 0; i < m_nInput; i++){
        grads(i) = (net.inputLayer(i) - net.inputBias(i))/m_sigma2;
    }

    int k = 0;
    //derivative wrt. hidden bias
    for(int i = m_nInput; i < (m_nInput+m_nHidden); i++){
        grads(i) = 1/(exp_neg_Q(k) + 1);
        k++;
    }


    //derivative wrt weights
    k = m_nInput+m_nHidden;
    for(int i = 0; i < m_nVisible; i++){
        for(int j = 0; j < m_nHidden; j++){
            // grads(k) = net.inputLayer(i)/(sig2Inv*(Q(j) + 1));
            grads(k) = net.inputLayer(i)/(m_sigma2*(exp_neg_Q(j) + 1));
            k++;
        }
    }

    return grads;

}







//
