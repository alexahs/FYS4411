#pragma once
#include "hamiltonian.h"


class Optimizer {
private:
    double m_eta;
    int m_whichMethod;
    int m_nOptimizeIters;

public:
    Optimizer(double eta, int whichMethod, int nOptimizeIters);


    void optimize();
};
