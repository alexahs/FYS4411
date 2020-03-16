#include "uniformlattice.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "Math/random.h"
#include "Misc/particle.h"
#include "Misc/system.h"

using std::cout;
using std::endl;

UniformLattice::UniformLattice(System*    system,
                             int        numberOfDimensions,
                             int        numberOfParticles,
                             double     characteristicLength,
                             double     hardSphereRadius)  :
        InitialState(system) {
    assert(numberOfDimensions == 3 && numberOfParticles > 0);
    assert(hardSphereRadius > 0);
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParticles  = numberOfParticles;
    m_characteristicLength = characteristicLength;
    m_hardShpereRadius = hardSphereRadius;



    m_system->setNumberOfDimensions(numberOfDimensions);
    m_system->setNumberOfParticles(numberOfParticles);
    setupInitialState();
}

void UniformLattice::setupInitialState() {
    /*
     attempt at initializing particles in a lattice
    */

    double l = m_hardShpereRadius*10;


    //formula for computing side length of lattice can be improved..
    int sideLength = floor(pow(m_numberOfParticles, 1.0/3.0)) + 1;

    std::vector<std::vector<double>> latticePoint = std::vector<std::vector<double>>();

    int nPoints = 0;
    //creates a few more points than nessecary...
    for(int x = -sideLength/2; x <= sideLength/2; x++){
        cout << x << endl;
        for(int y = -sideLength/2; y <= sideLength/2; y++){
            for(int z = -sideLength/2; z <= sideLength/2; z++){

                std::vector<double> position = {x*l, y*l, z*l};

                latticePoint.push_back(position);
                nPoints++;
            }
        }
    }


    std::vector<int> indices = std::vector<int>();

    for(int i = 0; i < m_numberOfParticles; i++){
        indices.push_back(i);
    }

    assert(nPoints >= m_numberOfParticles);


    for(int i = 0; i < m_numberOfParticles; i++){
        m_particles.push_back(new Particle());
        m_particles.at(i)->setNumberOfDimensions(m_numberOfDimensions);
        int idx = Random::nextInt(m_numberOfParticles-i); //random index
        m_particles.at(i)->setPosition(latticePoint[i]);
        m_particles.at(i)->setPosition(latticePoint[indices[idx]]);
        indices.erase(indices.begin() + idx);

    }

    // for(int i = 0; i < m_numberOfParticles; i++){
    //     for(int j = 0; j < 3; j++){
    //         cout << m_particles.at(i)->getPosition()[j] << "  ";
    //     }
    //     cout << endl;
    // }

    cout << nPoints << " semi uniform lattice points created" << endl;
    // cout << count << endl;
    // cout << m_numberOfParticles << endl;
}
