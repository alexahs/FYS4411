#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

class Particle
{
public:
    Particle();
    void setPosition(const std::vector<double> &position);
    void adjustPosition(double change, int dimension);
    void setNumberOfDimensions(int m_numberOfDimensions);
    std::vector<double> getPosition() {return m_position;}

private:
    int m_numberOfDimensions = 0;
    std::vector<double> m_position = std::vector<double>();
};


#endif //PARTICLE_H
