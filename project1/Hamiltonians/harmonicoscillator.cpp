#include "harmonicoscillator.h"
#include "hamiltonian.h"
// #include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"
#include <cassert>

HarmonicOscillator::HarmonicOscillator(double omega){
    m_omega = omega;
}
