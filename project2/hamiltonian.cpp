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
    /*
    Method for computing the local energy of the Neural Quantum State
    */
    double localEnergy = 0;
    for(int m = 0; m < m_nInput; m++){
        Eigen::VectorXd derivatives = nqs.computeFirstAndSecondDerivatives(m);
        double dx = derivatives(0);
        double ddx = derivatives(1);
        double xSquared = nqs.net.inputLayer(m)*nqs.net.inputLayer(m);
        localEnergy += -dx*dx - ddx + m_omega*m_omega*xSquared;
    }

    localEnergy *= 0.5;

    //Compute the Jastrow factor if particles are interacting
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
