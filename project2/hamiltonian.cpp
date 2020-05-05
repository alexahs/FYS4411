#include "hamiltonian.h"
#include <vector>


Hamiltonian::Hamiltonian(double omega, int nParticles, int nDims, bool interaction){
    m_omega = omega;
    m_nParticles = nParticles;
    m_nDims = nDims;
    m_interaction = interaction;
}

double Hamiltonian::computeLocalEnergy(NeuralQuantumState &nqs){
    // equation 114 in lecture notes

    int nNodes = m_nParticles*m_nDims;
    double localEnergy = 0;

    for(int m = 0; m < nNodes; m++){
        std::vector<double> derivatives = nqs.computeFirstAndSecondDerivatives(m);
        double dx = derivatives[0];
        double ddx = derivatives[1];
        double x2 = nqs.m_inputLayer[m]*nqs.m_inputLayer[m];

        localEnergy += -1*dx*dx - ddx + m_omega*m_omega*x2;
    }

    localEnergy *= 0.5;

    if (m_interaction) {
        double distances = 0;
        for(int p = 0; p < m_nParticles; p++){
            for(int q = p; q < m_nParticles; q++){
                distances += nqs.computeDistance(p, q);
            }
        }

        localEnergy += 1/distances;
    }

    return localEnergy;
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
