#include "sampler.h"
#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "optimizer.h"
#include "random.h"


Sampler::Sampler(int nMCcycles,
                 int samplingRule,
                 double tolerance,
                 Hamiltonian &hamiltonian,
                 NeuralQuantumState &nqs,
                 Optimizer &optimizer) :
    m_hamiltonian(hamiltonian), m_nqs(nqs), m_optimizer(optimizer) {
    m_nMCcycles = nMCcycles;
    m_samplingRule = samplingRule;
    m_tolerance = tolerance;
    m_hamiltonian = hamiltonian;
    m_nqs = nqs;
    m_optimizer = optimizer;

    m_nDims = nqs.getNumberOfDims();
    m_nParticles = nqs.getNumberOfParticles();
    m_nHidden = nqs.getNumberOfHidden();
    m_nInput = nqs.getNumberOfInputs();

}

bool Sampler::metropolisStep(int particleNumber){
    std::vector<double> positionOld = m_nqs.m_inputLayer;
    int idxStart = particleNumber*m_nDims;
    int idxStop = idxStart + m_nDims;

    std::vector<double> proposedStep;
    for(int node = idxStart; node < idxStop; node++){
        double step = Random::nextDouble() - 0.5;
        proposedStep.push_back(step);
        m_nqs.adjustPosition(node, step);
    }

    double wfNew = m_nqs.evaluate();
    double ratio = wfNew*wfNew/(m_wfOld*m_wfOld);

    if(Random::nextDouble() <= ratio){
        m_wfOld = wfNew;
        return true;
    }
    else{
        int i = 0;
        for(int node = idxStart; node < idxStop; node++){
            m_nqs.adjustPosition(node, - proposedStep[i]);
            i++;
        }
        return false;
    }
}


void Sampler::runSampling(){
    /*
    * pretty much the same as in project 1, main flow:
    * loop over nuber of cycles:
    *     loop over particles:
    *         calculate new position (accorting to some rule, probably qFroce)
    *         calculate new values of wavefunction (m_nqs.evaluate())
    *         calculate ratio of WFs (either by standard metropolis, importance or gibbs sampling)
              if random() <= ratio:
                  update positions and WF

          calculate local energy and add to cumulative energy
          calculate cost gradient and add to cumulative gradient

     take averages of cumulative sums (save to member variables possibly)
    */

    m_dPsiOld = m_nqs.m_grads;

    m_wfOld = m_nqs.evaluate();
    m_localEnergyOld = m_hamiltonian.computeLocalEnergy(m_nqs);
    double localEnergy = 0;
    s_dPsi dPsi;

    for(int cycle = 0; cycle < m_nMCcycles; cycle++){
        for(int particle = 0; particle < m_nParticles; particle++){
            if(metropolisStep(particle)) {
                m_acceptedSteps++;
                localEnergy = m_hamiltonian.computeLocalEnergy(m_nqs);
                m_localEnergyOld = localEnergy;
                dPsi = m_nqs.computeCostGradient();
                m_dPsiOld = dPsi;
            }
            else{
                localEnergy = m_localEnergyOld;
                dPsi = m_dPsiOld;

            }
        }

        m_energy += localEnergy;
        m_energy2 += localEnergy*localEnergy;

        for(int i = 0; i < m_nInput; i++){
            m_dPsiFinal.dInputBias[i] += dPsi.dInputBias[i];
            for(int j = 0; j < m_nHidden; j++){
                m_dPsiFinal.dWeights[i][j] += dPsi.dWeights[i][j];
            }
        }

        for(int i = 0; i < m_nHidden; i++){
            m_dPsiFinal.dHiddenBias[i] += dPsi.dHiddenBias[i];
        }

    }// end MC cycles

    m_energy /= m_nMCcycles;
    m_energy2 /= m_nMCcycles;

    for(int i = 0; i < m_nInput; i++){
        m_dPsiFinal.dInputBias[i] /= m_nMCcycles;
        for(int j = 0; j < m_nHidden; j++){
            m_dPsiFinal.dWeights[i][j] /= m_nMCcycles;
        }
    }

    for(int i = 0; i < m_nHidden; i++){
        m_dPsiFinal.dHiddenBias[i] /= m_nMCcycles;
    }


}

void Sampler::runOptimization(){
    /*
    * main flow:
    * loop max gradient descent iters:
    *     runSampling() to get energy and weight gradients
    *     update weights by gradient descent (or something a bit more sophisticated)
    *     save energy to file/memory
    *     if "global" minimum reached:
    *         break
    */
    int a = 1;
}
