#include "initialstate.h"

InitialState::InitialState(System* system) {
    m_system = system;
}

void InitialState::setCharacteristicLength(double a0) {
    m_characteristicLength = a0;
}
