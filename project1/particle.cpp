#include "particle.h"
#include <cassert>


Particle::Particle()
{
    std::vector<double> initialPosition(1);
    initialPosition.assign(1, 0.0);
    setNumberOfDimensions(1);
    setPosition(initialPosition);
}

void Particle::setPosition(const std::vector<double> &position)
{
    assert(position.size() == m_numberOfDimensions);
    m_position = position;
}

void Particle::adjustPosition(double change, int dimension)
{
    m_position.at(dimension) += change;
}

void Particle::setNumberOfDimensions(int numberOfDimensions)
{
    m_numberOfDimensions = numberOfDimensions;
}
