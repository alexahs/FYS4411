#pragma once

#include "system.h"

class MetropolisTest{
public:
    MetropolisTest(class System* system);
    virtual bool metropolisTest() = 0;

protected:
    System* m_system = nullptr;
};
