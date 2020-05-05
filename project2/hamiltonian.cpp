#include "hamiltonian.h"
#include <vector>


Hamiltonian::Hamiltonian(double omega){
    m_omega = omega;
}

double Hamiltonian::computeLocalEnergy(){
    // equation 114 in lecture notes





    return 1.0;
}


std::vector<double> Hamiltonian::computeCostGradient(){
    //gradients wrt variational parameters (weights / biases)
    //equations 105, 106 and 107 in lecture notes

    //a bit unsure about this function, might have to call it
    //seperately for each gradient component, since the shapes mismatch
    double grad1 = 1;
    double grad2 = 1;
    double grad3 = 1;
    std::vector<double> gradients = {grad1, grad2, grad3};
    return gradients;
}


double Hamiltonian::evaluateCost(){
    //cost function wrt variational parameters to be used in gradient descent
    //equation 99 in lecture notes
}
