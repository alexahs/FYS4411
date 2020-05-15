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
                 double stepLength,
                 double timeStep,
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
    m_stepLength = stepLength;
    m_timeStepDiffusion = timeStep*0.5;
    m_sqrtTimeStep = sqrt(timeStep);

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
        double step = m_stepLength*(Random::nextDouble() - 0.5);
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

bool Sampler::sample(int particleNumber){
    if(m_samplingRule == 1){
        return metropolisStep(particleNumber);
    }
    if(m_samplingRule == 2){
        return importanceStep(particleNumber);
    }

}

bool Sampler::importanceStep(int particleNumber){
    int idxStart = particleNumber*m_nDims;
    int idxStop = idxStart + m_nDims;
    int m = idxStart;
    Eigen::VectorXd qForceOld(m_nDims);
    Eigen::VectorXd qForceNew(m_nDims);
    Eigen::VectorXd posOld(m_nDims);
    Eigen::VectorXd posNew(m_nDims);
    Eigen::VectorXd proposedStep(m_nDims);
    double sigma2 = m_nqs.getSigma2();


    //old quantities
    Eigen::VectorXd Q = m_nqs.computeQfactor();
    int k = 0;
    for(int i = idxStart; i < idxStop; i++){
        double sum = 0;
        for(int n = 0; n < m_nHidden; n++){
            sum += m_nqs.net.weights(m, n)/(exp(-Q(n)) + 1);
        }
        qForceOld(k) = 2/sigma2*(-(m_nqs.net.inputLayer(i) - m_nqs.net.inputBias(i)) + sum);
        posOld(k) = m_nqs.net.inputLayer(i);
        //calculate new position
        double step = qForceOld(k)*m_timeStepDiffusion + Random::nextGaussian(0, 1.0)*m_sqrtTimeStep;
        proposedStep(k) = step;
        m_nqs.adjustPosition(i, step);

        qForceNew(k) = 2/sigma2*(-(m_nqs.net.inputLayer(i) - m_nqs.net.inputBias(i)) + sum);


        k++;
    }

    //new quantities
    double wfNew = m_nqs.evaluate();
    Q = m_nqs.computeQfactor();
    k = 0;
    for(int i = idxStart; i < idxStop; i++){
        double sum = 0;
        for(int n = 0; n < m_nHidden; n++){
            sum += m_nqs.net.weights(m, n)/(exp(-Q(n)) + 1);
        }
        qForceNew(k) = 2/sigma2*(-(m_nqs.net.inputLayer(i) - m_nqs.net.inputBias(i)) + sum);
        posNew(k) = m_nqs.net.inputLayer(i);
        k++;
    }

    //compute greens function
    double greensRatio = 0;
    for(int i = 0; i < m_nDims; i++){
        double term1 = posOld(i) - posNew(i) - m_timeStepDiffusion*qForceNew(i);
        double term2 = posNew(i) - posOld(i) - m_timeStepDiffusion*qForceOld(i);
        greensRatio += term2*term2 - term1*term1;
    }

    greensRatio /= 4*m_timeStepDiffusion;
    greensRatio = exp(greensRatio);
    double probabilityRatio = greensRatio*wfNew*wfNew/(m_wfOld*m_wfOld);

    if(Random::nextDouble() <= probabilityRatio){
        m_wfOld = wfNew;
        return true;
    }
    else{
        int i = 0;
        for(int node = idxStart; node < idxStop; node++){
            m_nqs.adjustPosition(node, - proposedStep(i));
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
    m_acceptedSteps = 0;

    m_dPsi.fill(0);
    m_dPsiTimesE.fill(0);

    for(int equil = 0; equil < (int) 0.1*m_nMCcycles; equil++){
        for(int particle = 0; particle < m_nParticles; particle++){
            int rndParticle = Random::nextInt(m_nParticles);
            bool equilibrationSteps = sample(rndParticle);
        }
    }

    for(int cycle = 0; cycle < m_nMCcycles; cycle++){
        for(int particle = 0; particle < m_nParticles; particle++){
            int rndParticle = Random::nextInt(m_nParticles);
            bool stepAccepted = sample(rndParticle);
            if(stepAccepted) {
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

        printInfo(i);

    }

    // printFinalSystemInfo();

}


void Sampler::printInfo(int step){
    cout << setw(10) << step;
    cout << setw(15) << setprecision(6) << m_energy;
    cout << setw(15) << setprecision(6) << m_energy2;
    cout << setw(15) << setprecision(6) << m_variance;
    cout << setw(15) << setprecision(6) << m_costGradient.sum();
    cout << setw(10) << setprecision(15) << m_acceptedSteps/m_nMCcycles << endl;
}


void Sampler::printInitalSystemInfo(){
    cout << endl;
    cout << " -------- System info -------- " << endl;
    cout << " * Number of dimensions        : " << m_nDims << endl;
    cout << " * Number of particles         : " << m_nParticles << endl;
    cout << " * Number of hidden nodes      : " << m_nHidden << endl;
    cout << " * Number of Metropolis steps  : " << "10^" << log10(m_nMCcycles) << endl;
    cout << " * Number of optimization steps: " << m_nOptimizeIters << endl;
    cout << " * Number of parameters        : " << m_nHidden*m_nInput + m_nHidden + m_nInput << endl << endl;
    cout << "===== Step ====== Energy ====== Energy2 ====== Variance ====== Cost ====== ";
    cout << "Accept ratio =====" << endl << endl;
}
