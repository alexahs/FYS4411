#include "hamiltonian.h"
#include <vector>


Hamiltonian::Hamiltonian(double omega, bool interaction, NeuralQuantumState &nqs){
    m_omega = omega;
    m_interaction = interaction;
    m_nParticles = nqs.getNumberOfParticles();
    m_nDims = nqs.getNumberOfDims();
    m_nInput = nqs.getNumberOfInputs();

}

double Hamiltonian::computeLocalEnergy(NeuralQuantumState &nqs){
    // equation 114 in lecture notes

    double localEnergy = 0;

    for(int m = 0; m < m_nInput; m++){
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




double Hamiltonian::evaluateCost(){
    //cost function wrt variational parameters to be used in gradient descent
    //equation 99 in lecture notes
}
