#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/correlated.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/elliptic_harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "InitialStates/uniformlattice.h"
#include "Math/random.h"
#include "Misc/system.h"
#include "Misc/particle.h"
#include "Misc/sampler.h"
#include "Misc/wfsampler.h"
#include "Misc/writefile.h"

using namespace std;

/*
    - Implement gradient descent for correlated WF
    - Implement oneBody density
*/

// Run VMC for spherical HO trap
// void run_bruteforce_vmc(double alpha_min, double alpha_max, double alpha_step);
// void run_single_vmc(double alpha, int numberOfSteps);

// Correlated = Interacting Bosons. Two approaches for finding optimal alpha:
void correlated_brute_force(int numberOfParticles);
void correlated_gradient_descent(int numberOfParticles, double alpha);

int main(int argc, char** argv) {
    // NOTE: number of Metropolis steps must be a 2^N for blocking resampling to run
    if (argc < 2) {
        cout << "Usage: \n\t./vmc <number_of_particles>" << endl;
        cout << "Example: \n\t./vmc 100" << endl;
        return 1;
    }
    int nParticles = atoi(argv[1]);
    // Control which analysis to perform:
    // run_bruteforce_vmc(0.1, 0.9, 0.05);
    // run_gradient_descent(500, 0.2, 0.001);
    // run_single_vmc(0.5, pow(2, 18));
    // correlated_brute_force(nParticles);
    vector<double> alphas;
    for(double alpha = 0.4; alpha < 0.6; alpha+=0.025) alphas.push_back(alpha);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < alphas.size(); i++) correlated_gradient_descent(nParticles, alphas[i]);
    return 0;
}

void correlated_brute_force(int numberOfParticles) {
    /* This case is done for only:
     *      3 Dimensions,
     *      Metropolis sampling rule (and not importance sampling)
     *      Analytic double derivative (numerical algorithm is slower by ~ 3 times)
     * This is the brute-force method for finding the optimal variational parameter
     * which is only alpha.
     */
    // Fixed parameters (should not be changed)
    int numberOfDimensions         = 3;          // Dimensions
    double gamma                   = 2.82843;    // omega_z / omega_ho.
    double beta                    = gamma;
    double characteristicLength    = 2.5;        // Side length of box to initialize particles within
    double bosonDiameter           = 0.00433;    // Fixed as in refs.
    int numVarParameters           = 2;
    std::vector<std::vector<double>> tempEnergies;
    std::vector<double> allEnergies;
    // Tweakable parameters
    int numberOfSteps              = int (pow(2, 20)); // Metropolis steps
    double stepLength              = 0.1;        // Metropolis: step length
    double equilibration           = 0.1;        // Amount of the total steps used for equilibration.
    double alphaMin                = 0.2;
    double alphaMax                = 0.8;
    double alphaStep               = 0.1;
    int numAlphas                  = int ((alphaMax - alphaMin) / alphaStep + 1);
    // Inital print
    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps,
        equilibration, numVarParameters);
    // Main loop: brute-force over alphas
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    Random::setSeed(- 1 - omp_get_thread_num());
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<numAlphas; ++i) {
        double alpha = alphaMin + i*alphaStep;
        System* system = new System();
        // Please note that system by default uses
        // * Analytic double derivative
        // * Standard Metropolis sampling
        system->setNumberOfParticles         (numberOfParticles);
        system->setNumberOfDimensions        (numberOfDimensions);
        system->setSampler                   (new Sampler(system));
        system->setHamiltonian               (new EllipticHarmonicOscillator(
                                                    system,
                                                    gamma,
                                                    bosonDiameter));
        system->setInitialState              (new RandomUniform(
                                                    system,
                                                    numberOfDimensions,
                                                    numberOfParticles,
                                                    characteristicLength));
        system->setWaveFunction              (new Correlated(
                                                    system,
                                                    alpha,  // The only variational parameter which is varied
                                                    beta, // gamma = beta = 2.82843
                                                    bosonDiameter));
        system->setEquilibrationFraction     (equilibration);
        system->setStepLength                (stepLength);
        system->setNumberOfMetropolisSteps   (numberOfSteps);
        system->runMetropolisSteps           ();
        vector<double> energySamples = system->getSampler()->getEnergySamples();
        writeFileEnergy(energySamples, numberOfDimensions, numberOfParticles, numberOfSteps,
             "correlated_bruteforce/alpha_" + to_string(alpha).substr(0, 5));
        // cout << "alpha = " << alpha << " completed.\n";
    } // End parallel
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
}

void correlated_gradient_descent(int numberOfParticles, double alpha) {
    /* This case is done for only:
     *      3 Dimensions,
     *      Metropolis sampling rule (and not importance sampling)
     *      Analytic double derivative (numerical algorithm is slower by ~ 3 times)
     * This is the gradiet descent method for finding the optimal variational parameter
     * which is only alpha.
     */
    // Fixed parameters (should not be changed)
    int numberOfDimensions         = 3;          // Dimensions
    double gamma                   = 2.82843;    // omega_z / omega_ho.
    double beta                    = gamma;
    double characteristicLength    = 2.5;        // Side length of box to initialize particles within
    double bosonDiameter           = 0.00433;    // Fixed as in refs.
    int numVarParameters           = 2;
    double cost;
    // Tweakable parameters
    int numberOfSteps              = int (pow(2, 17)); // Metropolis steps per alpha
    double stepLength              = 0.1;        // Metropolis: step length
    double equilibration           = 0.1;        // Amount of the total steps used for equilibration.
    double tol                     = 1e-5;
    // double alpha                   = 0.5;        // Initial Alpha, educated guess based on brute-force method
    double learningRate            = 0.01;
    int iter                       = 0;
    int maxIter                    = 50;
    std::vector<double> alphaVec;
    // Inital print
    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps,
        equilibration, numVarParameters);
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    double decay = 0.01;
    do {
        alphaVec.push_back(alpha);     // Save alpha
        // Please note that system by default uses
        // * Analytic double derivative
        // * Standard Metropolis sampling
        System* system = new System();
        system->setNumberOfParticles         (numberOfParticles);
        system->setNumberOfDimensions        (numberOfDimensions);
        system->setSampler                   (new WfSampler(system));
        system->setHamiltonian               (new EllipticHarmonicOscillator(
                                                    system,
                                                    gamma,
                                                    bosonDiameter));
        system->setInitialState              (new RandomUniform(
                                                    system,
                                                    numberOfDimensions,
                                                    numberOfParticles,
                                                    characteristicLength));
        system->setWaveFunction              (new Correlated(
                                                    system,
                                                    alpha,  // The only variational parameter which is varied
                                                    beta,   // gamma = beta = 2.82843
                                                    bosonDiameter));
        system->setEquilibrationFraction     (equilibration);
        system->setStepLength                (stepLength);
        system->setNumberOfMetropolisSteps   (numberOfSteps);
        system->runMetropolisSteps           ();

        // Get cost
        learningRate /= 1 + decay*iter;
        counter++;
        cost = system->getWaveFunction()->evaluateCostFunction();
        alpha -= learningRate*cost;           // Compute new alpha with GD
        cout << "Iteration " << ++iter << ", alpha = ";
        cout << fixed << setprecision(12) << alpha;
        cout << "Learning Rate = " << learningRate;
        cout << ", cost = " << cost << endl;
    } while (abs(alpha - alphaVec[iter-1]) > tol && iter < maxIter);
    // Print timing results
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    if (iter < maxIter) {
        cout << "Done. Converged in " << iter << " steps" << endl;
    } else {
        cout << "Done. Did not converge within tol = " << tol << " in "  << maxIter << " steps" << endl;
    }
    writeFileAlpha(alphaVec, numberOfParticles, numberOfSteps);
}

void run_single_vmc(double alpha, int numberOfSteps) {
    int numberOfDimensions         = 3;         // Dimensions
    int numberOfParticles          = 100;       // Particles in system
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    bool importanceSampling        = true;      // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = false;     // Otherwise: use analytical expression for 2nd derivative

    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();


    int nProcs = omp_get_num_procs();
    int stepsPerProc = numberOfSteps/nProcs;
    std::vector<std::vector<double>> tempEnergies;
    std::vector<double> allEnergies;

    #pragma omp parallel for schedule(dynamic)
        for(int i=0; i < nProcs; i++) {
            System* system = new System();
            system->setSampler                  (new Sampler(system));
            system->setHamiltonian              (new HarmonicOscillator(system, omega));
            system->setWaveFunction             (new SimpleGaussian(system, alpha));
            system->setInitialState             (new RandomUniform(system,
                                                        numberOfDimensions,
                                                        numberOfParticles,
                                                        characteristicLength));

            system->setEquilibrationFraction     (equilibration);
            system->setStepLength                (stepLength);
            system->setNumberOfMetropolisSteps   (numberOfSteps);
            system->setImportanceSampling        (importanceSampling, timeStep);
            system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
            system->runMetropolisSteps           ();
            vector<double> energySamples = system->getSampler()->getEnergySamples();
            #pragma omp critial
            tempEnergies.push_back(energySamples);
        }//end parallel


    for(int i = 0; i < nProcs; i++){
        allEnergies.insert(allEnergies.end(),
        tempEnergies[i].begin(),
        tempEnergies[i].end());
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    writeFileEnergy(allEnergies, numberOfDimensions, numberOfParticles, numberOfSteps, "simplegaussian/");
}

void run_bruteforce_vmc(double alpha_min, double alpha_max, double alpha_step) {

    int numberOfDimensions         = 3;         // Dimensions
    int numberOfParticles          = 10;        // Particales in system
    int numberOfSteps              = (int) 10; // Monte Carlo cycles
    double omega                   = 1.0;       // Oscillator frequency.
    double stepLength              = 1.0;       // Metropolis: step length
    double timeStep                = 0.01;      // Metropolis-Hastings: time step
    double h                       = 0.001;     // Double derivative step length
    double equilibration           = 0.1;       // Amount of the total steps used for equilibration.
    double characteristicLength    = 1.0;       // a_0: natural length scale of the system
    bool importanceSampling        = true;     // Otherwise: normal Metropolis sampling
    bool numericalDoubleDerviative = false;     // Otherwise: use analytical expression for 2nd derivative
    bool saveEnergySamples         = true;
    // Initialize vectors where results will be stored
    vector<double> alphaVec;
    for(double alpha=alpha_min; alpha<=alpha_max; alpha+=alpha_step) { alphaVec.push_back(alpha); }
    vector<double> energyVec(alphaVec.size(), 0);
    vector<double> energy2Vec(alphaVec.size(), 0);
    vector<double> varianceVec(alphaVec.size(), 0);
    vector<double> acceptRatioVec(alphaVec.size(), 0);
    printInitalSystemInfo(numberOfDimensions, numberOfParticles, numberOfSteps, equilibration, 1);

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<alphaVec.size(); i++) {
            System* system = new System();
            system->setSampler                  (new Sampler(system));
            system->setHamiltonian              (new HarmonicOscillator(system, omega));
            system->setWaveFunction             (new SimpleGaussian(system, alphaVec.at(i)));
            system->setInitialState             (new RandomUniform(system,
                                                        numberOfDimensions,
                                                        numberOfParticles,
                                                        characteristicLength));

            system->setEquilibrationFraction     (equilibration);
            system->setStepLength                (stepLength);
            system->setNumberOfMetropolisSteps   (numberOfSteps);
            system->setImportanceSampling        (importanceSampling, timeStep);
            system->setNumericalDoubleDerivative (numericalDoubleDerviative, h);
            system->runMetropolisSteps           ();

            Sampler* system_sampler = system->getSampler();
            energyVec.at(i) = (system_sampler->getEnergy());
            energy2Vec.at(i) = (system_sampler->getEnergy2());
            varianceVec.at(i) = (system_sampler->getVariance());
            acceptRatioVec.at(i) = (system_sampler->getAcceptRatio());
        }
        //end parallel region

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    printFinal(1, chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    writeFileOneVariational(numberOfDimensions, numberOfParticles, numberOfSteps,
      int (equilibration*numberOfSteps), numericalDoubleDerviative,
      alphaVec, energyVec, energy2Vec,
      varianceVec, acceptRatioVec);
}
