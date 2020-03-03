#pragma once
#include "Misc/particle.h"
#include "Misc/system.h"
#include "metropolistest.h"

class StandardMetropolisTest : public MetropolisTest {
public:
    StandardMetropolisTest(class System* system);
    bool metropolisTest();
};
