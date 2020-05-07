#include "sampler.h"
#include "hamiltonian.h"
#include "neuralquantumstate.h"
#include "optimizer.h"
#include "random.h"
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::setw;
using std::setprecision;


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

    //store gradient vector in one long 1d vec
    m_dPsi.resize(m_nInput + m_nHidden + m_nInput*m_nHidden);
    m_dPsiTimesE.resize(m_nInput + m_nHidden + m_nInput*m_nHidden);

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

    Eigen::VectorXd netGrads1d = m_nqs.computeCostGradient();
    m_wfOld = m_nqs.evaluate();
    double localEnergy = m_hamiltonian.computeLocalEnergy(m_nqs);

    m_energy = 0;
    m_energy2 = 0;

    m_dPsi.fill(0);
    m_dPsiTimesE.fill(0);

    for(int cycle = 0; cycle < m_nMCcycles; cycle++){
        for(int particle = 0; particle < m_nParticles; particle++){
            if(metropolisStep(particle)) {
                m_acceptedSteps++;
                localEnergy = m_hamiltonian.computeLocalEnergy(m_nqs);
                netGrads1d = m_nqs.computeCostGradient();
            }
        }

        m_energy += localEnergy;
        m_energy2 += localEnergy*localEnergy;
        m_dPsi += netGrads1d;
        m_dPsiTimesE += netGrads1d*localEnergy;


    }// end MC cycles

    m_energy /= m_nMCcycles;
    m_energy2 /= m_nMCcycles;
    m_dPsi /= m_nMCcycles;
    m_dPsiTimesE /= m_nMCcycles;

    m_variance = m_energy2 - m_energy*m_energy;
    m_costGradient = 2*(m_dPsiTimesE - m_energy*m_dPsi);




}

void Sampler::runOptimization(){
    /*
    Optimize weights and biases after each set of MC runs
    */
    printInitalSystemInfo();
    for(int i = 0; i < m_nOptimizeIters; i++){
        runSampling();
        m_optimizer.optimize(m_nqs, m_costGradient, m_nInput, m_nHidden);

        printInfo();
    }

}

void Sampler::printInfo(){
    cout << setw(13) << setprecision(5) << m_energy;
    cout << setw(14) << setprecision(5) << m_energy2;
    cout << setw(16) << setprecision(5) << m_variance << endl;
}


void Sampler::printInitalSystemInfo(){
    cout << endl;
    cout << " -------- System info -------- " << endl;
    cout << " * Number of dimensions        : " << m_nDims << endl;
    cout << " * Number of particles         : " << m_nParticles << endl;
    cout << " * Number of hidden layers     : " << m_nHidden << endl;
    cout << " * Number of Metropolis steps  : " << m_nMCcycles << endl;
    cout << " * Number of optimization steps: " << m_nOptimizeIters << endl;
    cout << " * Number of parameters        : " << m_nHidden*m_nInput << endl << endl;
    cout << "====== Energy ====== Energy2 ====== Variance ======" << endl << endl;
}
