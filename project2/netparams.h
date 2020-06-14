#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct NetParams {
    /*
    Data struct containing the parameters of the NQS
    */
    VectorXd inputLayer;
    VectorXd hiddenLayer;
    VectorXd inputBias;
    VectorXd hiddenBias;
    MatrixXd weights;

    int inputSize;
    int hiddenSize;

    NetParams(){}

    NetParams(int nInput, int nHidden){
        inputSize = nInput;
        hiddenSize = nHidden;
        inputLayer.resize(nInput);
        inputBias.resize(nInput);

        hiddenLayer.resize(nHidden);
        hiddenBias.resize(nHidden);
        weights.resize(nInput, nHidden);
    }
};
