#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::setw;
using std::setprecision;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct NetParams {
    VectorXd inputLayer;
    VectorXd hiddenLayer;
    VectorXd inputBias;
    VectorXd hiddenBias;
    MatrixXd weights;

    VectorXd dInputBias;
    VectorXd dHiddenBias;
    MatrixXd dWeights;

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

        dInputBias.resize(nInput);
        dHiddenBias.resize(nHidden);

        weights.resize(nInput, nHidden);
        dWeights.resize(nInput, nHidden);
    }



};
