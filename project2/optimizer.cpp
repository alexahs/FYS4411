#include "optimizer.h"
#include <vector>



Optimizer::Optimizer(double eta, int whichMethod, int nOptimizeIters){
    m_eta = eta;
    m_whichMethod = whichMethod;
    m_nOptimizeIters = nOptimizeIters;
}



void Optimizer::optimize(){
    //gradient denscent
    if (m_whichMethod == 1) {
        int a = 1;


    }
    if (m_whichMethod == 2){
        //some other optim method
        int a = 1;
    }
}
