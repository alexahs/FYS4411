#include "optimizer.h"
#include "netparams.h"
#include <vector>
#include <Eigen/Dense>

Optimizer::Optimizer(double eta, int whichOptimizer){
    m_eta = eta;
    m_whichOptimizer = whichOptimizer;
}

void Optimizer::optimize(NeuralQuantumState &nqs, Eigen::VectorXd grads, int nInput, int nHidden){
    /*
    Method for updating the parameters of the wave function by simple gradient descent.
    */

    //gradient denscent
    if (m_whichOptimizer == 1) {

        for(int i = 0; i < nInput; i++){
            nqs.net.inputBias(i) -= m_eta*grads(i);
        }

        int k = nInput;
        for(int i = 0; i < nHidden; i++){
            nqs.net.hiddenBias(i) -= m_eta*grads(k);
            k++;
        }

        k = nInput + nHidden;
        for(int i = 0; i < nInput; i++){
            for(int j = 0; j < nHidden; j++){
                nqs.net.weights(i, j) -= m_eta*grads(k);
                k++;
            }
        }
    }

    //Plans for a more sophisticated minimizing scheme...
    if (m_whichOptimizer == 2){}
}
