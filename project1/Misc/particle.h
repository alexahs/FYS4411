#pragma once
#include <vector>

class Particle {
    /* Particle class is how one single particle is defined.
     * We have the opportunity to easilty move it by 'adjustPosition()', or 
     * call the position vector by 'getPosition()'
     */
public:
    Particle();
    void setPosition(const std::vector<double> &position);
    void adjustPosition(double change, int dimension);
    void setNumberOfDimensions(int numberOfDimensions);
    std::vector<double> getPosition() { return m_position; }

private:
    int     m_numberOfDimensions = 0;
    std::vector<double> m_position = std::vector<double>();
};
