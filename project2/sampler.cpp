#include "sampler.h"
#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "optimizer.h"


Sampler::Sampler(int nMCcycles,
                 int nOptimizeIters,
                 int samplingRule,
                 double tolerance,
                 Hamiltonian &hamiltonian,
                 NeuralQuantumState &nqs,
                 Optimizer &optimizer) :
    m_hamiltonian(hamiltonian), m_nqs(nqs), m_optimizer(optimizer) {
    m_nMCcycles = nMCcycles;
    m_nOptimizeIters = nOptimizeIters;
    m_samplingRule = samplingRule;
    m_tolerance = tolerance;
    m_hamiltonian = hamiltonian;
    m_nqs = nqs;
    m_optimizer = optimizer;
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
    int a = 1;

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
