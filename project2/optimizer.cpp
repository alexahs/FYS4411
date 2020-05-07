#include "optimizer.h"
#include "netparams.h"
#include <vector>



Optimizer::Optimizer(double eta, int whichOptimizer){
    m_eta = eta;
    m_whichOptimizer = whichOptimizer;
}



void Optimizer::optimize(NeuralQuantumState &nqs){
    //gradient denscent
    if (m_whichOptimizer == 1) {
        std::cout << "GRADIENT DESCENT" << std::endl;
        nqs.net.inputBias -= m_eta*nqs.net.dInputBias;
        nqs.net.hiddenBias -= m_eta*nqs.net.dHiddenBias;
        nqs.net.weights -= m_eta*nqs.net.dWeights;

    }
    if (m_whichOptimizer == 2){
        //some other optim method
        int a = 1;
    }
}
