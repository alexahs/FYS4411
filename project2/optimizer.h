#pragma once


class Optimizer {
private:
    double m_eta;
    int m_whichMethod;

public:
    Optimizer(double eta, int whichMethod);

    void gradientDescent();

    void optimize();
};
