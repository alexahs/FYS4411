#pragma once
#include <vector>
#include <iomanip>

class System {
public:

    bool metropolisStep             ();
    bool importanceStep             ();
    void runMetropolisSteps         (int numberOfMetropolisSteps);
    void setNumberOfParticles       (int numberOfParticles);
    void setNumberOfDimensions      (int numberOfDimensions);
    void setStepLength              (double stepLength);
    void setEquilibrationFraction   (double equilibrationFraction);
    void setHamiltonian             (class Hamiltonian* hamiltonian);
    void setWaveFunction            (class WaveFunction* waveFunction);
    void setInitialState            (class InitialState* initialState);
    void setImportanceSampling      (bool importanceSampling, double timeStep);
    void setNumericalDoubleDerivative (bool numericalDoubleDerviative, double h);
    void setSampler                 (class Sampler* sampler);
    class WaveFunction*             getWaveFunction()   { return m_waveFunction; }
    class Hamiltonian*              getHamiltonian()    { return m_hamiltonian; }
    class Sampler*                  getSampler()        { return m_sampler; }
    std::vector<class Particle*>    getParticles()      { return m_particles; }
    int getNumberOfParticles()          { return m_numberOfParticles; }
    int getNumberOfDimensions()         { return m_numberOfDimensions; }
    int getNumberOfMetropolisSteps()    { return m_numberOfMetropolisSteps; }
    double getEquilibrationFraction()   { return m_equilibrationFraction; }
    bool getNumericalDoubleDerivative() { return m_numericalDoubleDerivative; }
    double getStepLength()              { return m_h; }
    double getSumRiSquared          ();

private:
    bool                            m_importanceSampling = false;
    bool                            m_numericalDoubleDerivative = false;
    int                             m_numberOfParticles = 0;
    int                             m_numberOfDimensions = 0;
    int                             m_numberOfMetropolisSteps = 0;
    double                          m_equilibrationFraction = 0.0;
    double                          m_stepLength = 0.0;
    double                          m_timeStep = 0.0;
    double                          m_h = 0.0;
    double                          m_timeStepDiffusion = 0.0;
    double                          m_sqrtTimeStep = 0.0;
    double                          m_invFourTimeStepDiffusion = 0.0;
    class WaveFunction*             m_waveFunction = nullptr;
    class Hamiltonian*              m_hamiltonian = nullptr;
    class InitialState*             m_initialState = nullptr;
    class Sampler*                  m_sampler = nullptr;
    std::vector<class Particle*>    m_particles = std::vector<class Particle*>();

    //temporary storage of the wavefunction in metropolis steps
    double                          wfOld = 0;

};
