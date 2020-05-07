#include "sampler.h"
#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "optimizer.h"
#include "random.h"
#include <Eigen/Dense>

using std::cout;
using std::endl;


Sampler::Sampler(int nMCcycles,
                 int samplingRule,
                 double tolerance,
                 int nOptimizeIters,
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
    m_nOptimizeIters = nOptimizeIters;

    m_nDims = nqs.getNumberOfDims();
    m_nParticles = nqs.getNumberOfParticles();
    m_nHidden = nqs.getNumberOfHidden();
    m_nInput = nqs.getNumberOfInputs();

    m_finalGradients = NetParams(m_nInput, m_nHidden);

}

bool Sampler::metropolisStep(int particleNumber){
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

    NetParams m_netOld = m_nqs.net;
    m_nqs.computeCostGradient();
    m_wfOld = m_nqs.evaluate();
    m_localEnergyOld = m_hamiltonian.computeLocalEnergy(m_nqs);
    double localEnergy = 0;

    m_energy = 0;
    m_energy2 = 0;
    m_finalGradients.dInputBias.fill(0);
    m_finalGradients.dHiddenBias.fill(0);
    m_finalGradients.dWeights.fill(0);


    for(int cycle = 0; cycle < m_nMCcycles; cycle++){
        for(int particle = 0; particle < m_nParticles; particle++){
            if(metropolisStep(particle)) {
                m_acceptedSteps++;
                localEnergy = m_hamiltonian.computeLocalEnergy(m_nqs);
                m_localEnergyOld = localEnergy;
                m_nqs.computeCostGradient();
            }
            else{
                localEnergy = m_localEnergyOld;
                m_nqs.net = m_netOld;

            }
        }

        m_energy += localEnergy;
        m_energy2 += localEnergy*localEnergy;

        m_finalGradients.dInputBias += m_nqs.net.dInputBias;
        m_finalGradients.dHiddenBias += m_nqs.net.dHiddenBias;
        m_finalGradients.dWeights += m_nqs.net.dWeights;

    }// end MC cycles

    m_energy /= m_nMCcycles;
    m_energy2 /= m_nMCcycles;

    m_finalGradients.dInputBias /= m_nMCcycles;
    m_finalGradients.dHiddenBias /= m_nMCcycles;
    m_finalGradients.dWeights /= m_nMCcycles;

    cout << "Energy:   " << m_energy << endl;
    cout << "Energy^2: " << m_energy2 << endl;



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

    // std::cout << "GOT THIS FAR" << std::endl;
    // cout << m_nOptimizeIters << endl;
    for(int i = 0; i < m_nOptimizeIters; i++){
        // std::cout << "GOT THIS FAR" << std::endl;
        runSampling();
        m_optimizer.optimize(m_nqs);
    }
    // std::cout << "GOT THIS FAR" << std::endl;
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);
    // runSampling();
    // m_optimizer.optimize(m_nqs);


    // int i = 0;
    // while (i < m_nOptimizeIters){
    //     std::cout << "GOT THIS FAR" << std::endl;
    //     runSampling();
    //     m_optimizer.optimize(m_nqs);
    //     i++;
    // }





}
