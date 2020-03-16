#include "uniformlattice.h"
#include <iostream>
#include <cassert>
#include "Math/random.h"
#include "Misc/particle.h"
#include "Misc/system.h"

using std::cout;
using std::endl;

UniformLattice::UniformLattice(System*    system,
                             int        numberOfDimensions,
                             int        numberOfParticles,
                             double     characteristicLength)  :
        InitialState(system) {
    assert(numberOfDimensions > 0 && numberOfParticles > 0);
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParticles  = numberOfParticles;
    m_characteristicLength = characteristicLength;



    m_system->setNumberOfDimensions(numberOfDimensions);
    m_system->setNumberOfParticles(numberOfParticles);
    setupInitialState();
}

void UniformLattice::setupInitialState() {
    for (int i=0; i < m_numberOfParticles; i++) {
        std::vector<double> position = std::vector<double>();

        for (int j=0; j < m_numberOfDimensions; j++) {


            position.push_back((Random::nextDouble() - 0.5)*m_characteristicLength);

        }
        m_particles.push_back(new Particle());
        m_particles.at(i)->setNumberOfDimensions(m_numberOfDimensions);
        m_particles.at(i)->setPosition(position);
    }
}
