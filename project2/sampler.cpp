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

    m_energyVals.resize(m_nOptimizeIters);
    m_energy2Vals.resize(m_nOptimizeIters);
    m_varianceVals.resize(m_nOptimizeIters);
    m_acceptRatioVals.resize(m_nOptimizeIters);

    // if(m_samplingRule == 3){
    //     for(int i = 0; i < m_nInput; i++){
    //         m_uniform_real_distribution_H(i) = Random::nextDouble();
    //     }
    // }

}

bool Sampler::sample(int particleNumber){
    if(m_samplingRule == 1){
        return metropolisStep(particleNumber);
    }
    if(m_samplingRule == 2){
        return importanceStep(particleNumber);
    }
    if(m_samplingRule == 3){
        return gibbsStep();
    }
    else{
        cout << "Invalild sampling rule." << endl;
        cout << "Usage:" << endl;
        cout << "(1) - Standard metropolis sampling" << endl << "(2) - Importance sampling" << endl << "(3) - Gibbs sampling" << endl;
        exit(1);
    }

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

bool Sampler::gibbsStep(){

    for(int j = 0; j < m_nHidden; j++){
         double z = m_nqs.net.hiddenBias(j) + m_nqs.net.inputLayer.dot(m_nqs.net.weights.col(j))/m_nqs.getSigma2();
         m_nqs.net.hiddenLayer(j) = Random::nextDouble() < sigmoid(z);
    }

    for(int i = 0; i < m_nInput; i++){
        double meanPos = m_nqs.net.inputBias(i) + m_nqs.net.weights.row(i)*m_nqs.net.hiddenLayer;
        double posDistribution = Random::nextGaussian(meanPos, m_nqs.getSigma());
        m_nqs.net.inputLayer(i) = posDistribution;
    }

    return true;
}

double Sampler::sigmoid(double z){
    return 1.0/(1 + exp(-z));
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
    if(m_samplingRule != 3){m_wfOld = m_nqs.evaluate();}
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
        // for(int particle = 0; particle < m_nParticles; particle++){
        int rndParticle = Random::nextInt(m_nParticles);
        bool stepAccepted = sample(rndParticle);
        if(stepAccepted) {
            m_acceptedSteps++;
            localEnergy = m_hamiltonian.computeLocalEnergy(m_nqs);
            netGrads1d = m_nqs.computeCostGradient();
        }
        // }

        m_energy += localEnergy;
        m_energy2 += localEnergy*localEnergy;
        m_dPsi += netGrads1d;
        m_dPsiTimesE += netGrads1d*localEnergy;

        if(m_finalRun){
            m_energySamples(cycle) = localEnergy;
            for(int i = 0; i < m_nDims; i++) {
                m_positions(m_nDims*cycle+i) = m_nqs.net.inputLayer.data()[i];
            }
        }
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
    if(m_printOptimInfo){printInitalSystemInfo();}

    m_finalRun = false;
    for(int step = 0; step < m_nOptimizeIters; step++){
        runSampling();
        // cout << m_nqs.net.hiddenLayer << endl;
        // exit(1);
        m_optimizer.optimize(m_nqs, m_costGradient, m_nInput, m_nHidden);

        m_acceptRatio = (double)m_acceptedSteps/(double)m_nMCcycles;

        m_energyVals(step) = m_energy;
        m_energy2Vals(step) = m_energy2;
        m_varianceVals(step) = m_variance;
        m_acceptRatioVals(step) = m_acceptRatio;
        if(m_printOptimInfo){printInfo(step);}

    }
    writeCumulativeResults();

    // printFinalSystemInfo();

}

void Sampler::runDataCollection(int nMCcycles, bool saveToFile){
    /*
    final big run with optimized weights
    */
    m_nMCcycles = nMCcycles;
    m_finalRun = true;
    m_energySamples.resize(m_nMCcycles);
    m_positions.resize(m_nDims*m_nMCcycles);
    runSampling();
    m_acceptRatio = (double)m_acceptedSteps/(double)m_nMCcycles/(double)m_nParticles;

    if (saveToFile) {
      writeEnergySamples();
    }
}

void Sampler::writeEnergySamples(){
    std::string filename = "./Data/energy_samples_";
    filename.append(std::to_string(m_nParticles) + "p_");
    filename.append(std::to_string(m_nDims) + "d_");
    filename.append(std::to_string(m_nHidden) + "h_");
    filename.append(std::to_string(m_nMCcycles) + "cycles_");
    filename.append(std::to_string(m_samplingRule) + "s_");
    filename.append(std::to_string(m_optimizer.getLearningRate()) + "eta.bin");

    std::string filename_2 = "./Data/pos_samples_";
    filename_2.append(std::to_string(m_nParticles) + "p_");
    filename_2.append(std::to_string(m_nDims) + "d_");
    filename_2.append(std::to_string(m_nHidden) + "h_");
    filename_2.append(std::to_string(m_nMCcycles) + "cycles_");
    filename_2.append(std::to_string(m_samplingRule) + "s_");
    filename_2.append(std::to_string(m_optimizer.getLearningRate()) + "eta.bin");

    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    outfile.write(reinterpret_cast<const char*> (m_energySamples.data()),m_energySamples.size()*sizeof(double));
    outfile.close();
    if(m_printOptimInfo){cout << " * Energies written to " << filename << endl;}

    std::ofstream outfile_2;
    outfile_2.open(filename_2, std::ios::out | std::ios::binary | std::ios::trunc);
    outfile_2.write(reinterpret_cast<const char*> (m_positions.data()), m_positions.size()*sizeof(double));
    outfile_2.close();
    if(m_printOptimInfo){cout << " * Positions written to " << filename_2 << endl;}
}

void Sampler::printFinalInfo(){
    cout << endl;
    cout << "------- Final values with optimized parameters -------" << endl;
    cout << "===== Cycles ====== Energy ====== Energy2 ====== Variance ===== Accept ratio =====" << endl << endl;
    cout << setw(10) << "2^" << log2(m_nMCcycles);
    cout << setw(15) << setprecision(6) << m_energy;
    cout << setw(15) << setprecision(6) << m_energy2;
    cout << setw(15) << setprecision(6) << m_variance;
    cout << setw(15) << setprecision(6) << m_acceptRatio << endl;
}

void Sampler::printInfo(int step){
    cout << setw(10) << step;
    cout << setw(15) << setprecision(6) << m_energy;
    cout << setw(15) << setprecision(6) << m_energy2;
    cout << setw(15) << setprecision(6) << m_variance;
    cout << setw(15) << setprecision(6) << m_costGradient.sum();
    cout << setw(15) << setprecision(6) << m_acceptRatio << endl;
}

void Sampler::printInitalSystemInfo(){
    cout << endl;
    cout << " -------- System info -------- " << endl;
    cout << " * Number of dimensions         : " << m_nDims << endl;
    cout << " * Number of particles          : " << m_nParticles << endl;
    cout << " * Number of hidden nodes       : " << m_nHidden << endl;
    cout << " * Number of hidden node density: " << m_nHidden/m_nInput << endl;
    cout << " * Number of Metropolis steps   : " << "2^" << log2(m_nMCcycles) << endl;
    cout << " * Number of optimization steps : " << m_nOptimizeIters << endl;
    cout << " * Number of parameters         : " << m_nHidden*m_nInput + m_nHidden + m_nInput << endl << endl;
    cout << "===== Step ====== Energy ====== Energy2 ====== Variance ====== Cost ====== ";
    cout << "Accept ratio =====" << endl << endl;
}

void Sampler::writeCumulativeResults(){
    /*
    writes results from optimizing
    */

    // std::string dir = "./Data/results_" + std::to_string(m_optimizer.getLearningRate()) + "eta/";
    std::string filename = "./Data/rbm_cumulative_results_";
    filename.append(std::to_string(m_nParticles) + "p_");
    filename.append(std::to_string(m_nDims) + "d_");
    filename.append(std::to_string(m_nHidden) + "h_");
    filename.append(std::to_string(m_nMCcycles) + "cycles_");
    filename.append(std::to_string(m_optimizer.getLearningRate()) + "eta.csv");

    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    outfile << "Energy,Energy2,Variance,AcceptRatio" << endl;
    for(int i = 0; i < m_nOptimizeIters; i++){
        outfile << m_energyVals(i) << ",";
        outfile << m_energy2Vals(i) << ",";
        outfile << m_varianceVals(i) << ",";
        outfile << m_acceptRatioVals(i) << endl;
    }

    outfile.close();
    if(m_printOptimInfo){cout << " * Cumulative results written to " << filename << endl;}

}

void Sampler::printGridSearchInfo(int i, int j){
    if(i == 0 && j == 0){
        cout << "===== Eta ===== nHidden ===== Energy ===== Energy2 ===== Variance ===== Accept ratio =====" << endl << endl;
    }
    cout << setw(8) << setprecision(6) << m_optimizer.getLearningRate();
    cout << setw(15) << setprecision(6) << m_nHidden;
    cout << setw(15) << setprecision(6) << m_energy;
    cout << setw(15) << setprecision(6) << m_energy2;
    cout << setw(15) << setprecision(6) << m_variance;
    cout << setw(15) << setprecision(6) << m_acceptRatio << endl;
}
