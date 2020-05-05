#include "optimizer.h"



Optimizer::Optimizer(double eta, int whichMethod){
    m_eta = eta;
    m_whichMethod = whichMethod;
}


void Optimizer::gradientDescent(){
    //implement gradient descent
    int a = 1;
}


void Optimizer::optimize(){
    if (m_whichMethod == 1) {
        gradientDescent();
    }
    if (m_whichMethod == 2){
        //some other optim method
        int a = 1;
    }
}
