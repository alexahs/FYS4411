#include "neuralquantumstate.h"
#include "Math/random.h"
#include "netparams.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using std::cout;
using std::endl;

NeuralQuantumState::NeuralQuantumState(int nParticles,
                                       int nDims,
                                       int nHidden,
                                       double sigma,
                                       long seed,
                                       double sigma_init,
                                       int samplingRule){
    m_nVisible = (int) nParticles*nDims;
    m_nInput = (int) nParticles*nDims;
    m_nHidden = nHidden;
    m_nParticles = nParticles;
    m_nDims = nDims;
    m_sigma_init = sigma_init;
    m_sigma = sigma;
    m_sigma2 = sigma*sigma;
    m_sigma4 = m_sigma2*m_sigma2;
    net = NetParams(m_nInput, m_nHidden);
    if(samplingRule == 3){m_isGibbsSampling = true;}
    initialize();
}

void NeuralQuantumState::initialize(){
    /*
    Initializes weights and biases of the network and
    initial positions of the particles

    Weights are normally distributed around zero with low
    variance (typically use sigma_init = 0.001)

    Initial positions are uniformly distributed in the
    range [-0.5, 0.5]. For systems of many particles,
    this method of initializing should be altered.
    */
    for(int i = 0; i < m_nVisible; i++){
        net.inputBias(i) = Random::nextGaussian(0, m_sigma_init);
        net.inputLayer(i) = Random::nextDouble() - 0.5;
        for(int j = 0; j < m_nHidden; j++){
            net.weights(i, j) = Random::nextGaussian(0, m_sigma_init);
        }
    }

    for(int i = 0; i < m_nHidden; i++){
        net.hiddenBias(i) = Random::nextGaussian(0, m_sigma_init);
    }
}

void NeuralQuantumState::adjustPosition(int node, double change){
    /*
    Adjusts the position of a particle along
    a single dimension
    */

    net.inputLayer(node) += change;
}

double NeuralQuantumState::evaluate(){
    /*
    Method for evaluating the wave function
    */

    double psi1 = 0;
    double psi2 = 1;
    for(int i = 0; i < m_nVisible; i++){
        psi1 +=  (net.inputLayer(i)-net.inputBias(i))*(net.inputLayer(i)-net.inputBias(i));
    }
    psi1 = exp(-psi1/(2*m_sigma2));

    Eigen::VectorXd Q = computeQfactor();
    for(int j = 0; j<m_nHidden; j++){
        psi2 *= (1 + exp(Q(j)));
    }

    return psi1*psi2;
}

Eigen::VectorXd NeuralQuantumState::computeQfactor(){
    /*
    Method for computing the argument of the exponential function
    in the wave function, which is used many times throughout the program.
    */

    Eigen::VectorXd Qfactor = net.hiddenBias + (((1.0/m_sigma2)*net.inputLayer).transpose()*net.weights).transpose();
    return Qfactor;
}

Eigen::VectorXd NeuralQuantumState::computeFirstAndSecondDerivatives(int nodeNumber){
    /*
    Computes the laplacian and first derivative of the wave function
    for one particle along one dimension.
    returns vector of d/dx_m [ln(psi)] and d^2/dx_m^2 [ln(psi)].
    */

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
    if(m_isGibbsSampling){
        dx *= 0.5;
        ddx *= 0.5;
    }

    Eigen::VectorXd derivatives(2);
    derivatives(0) = dx;
    derivatives(1) = ddx;
    return derivatives;
}

double NeuralQuantumState::computeDistance(int p, int q){
    /*
    Computes the distance between two particles p and q
    */
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
    /*
    Method for computing the gradients of the local energy
    wrt. the weights and biases.

    Returns them in a vector of size [M + N + M*N],
    where M and N are the number of visible and hidden nodes respectively.


    Index 0 to N-1

    The M first entries are the gradients wrt. input biases.
    The next N entries are wrt. hidden biases.
    The last M*N are wrt. the weights
    */
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
            grads(k) = net.inputLayer(i)/(m_sigma2*(exp_neg_Q(j) + 1));
            k++;
        }
    }

    // divide by a factor 2 if using Gibbs sampling
    if(m_isGibbsSampling){grads *= 0.5;}
    return grads;
}
