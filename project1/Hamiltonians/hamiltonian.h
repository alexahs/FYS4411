#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <vector>
// #include "../system.h"

class Hamiltonian
{
public:
    // Hamiltonian(class System* system);
    virtual double computeLocalEnergy(std::vector<class Particle*> particles) = 0;

private:
    // class System* m_system = nullptr;
};




#endif // HAMILTONIAN_H
