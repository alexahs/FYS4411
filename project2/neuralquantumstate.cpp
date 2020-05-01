#include "neuralquantumstate.h"
#include "Math/random.h"


NeuralQuantumState::NeuralQuantumState(int nParticles, int nDims, int nHidden, double sigma){
    //
    m_nVisible = nParticles*nDims;
    m_nHidden = nHidden;
    m_sigma = sigma;
}


void NeuralQuantumState::initialize(){

    initializeWeights();
    initializePositions();
}


void NeuralQuantumState::initializeWeights(){
    //randomly initialize weights or something
    // for i in nParticles*nDims:
    //     inputBias[i] = ...
    //     for j in nHidden:
    //         m_weights[i, j] = ...

    //for i in nHidden:
    //    hiddenBias[i] = ...
    int a = 1;

}

void NeuralQuantumState::initializePositions(){
    //initialize positions
    // for i in nParticles*nDims:
    //     m_inputLayer[i] = ...
    int a = 1;
}


double NeuralQuantumState::evaluate(){
    //evaluate the wavefunction
    double wf = 1;
    return wf;
}

double NeuralQuantumState::computeNabla2(){
    double gradient = 1;
    return gradient;
}

std::vector<double> NeuralQuantumState::computeQuantumForce(){
    std::vector<double> qForce = {1, 2, 3};
    return qForce;
}
