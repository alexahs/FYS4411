#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>

using std::cout;
using std::endl;
using std::setw;
using std::setprecision;

struct NetParams {
    std::vector<double> inputLayer;
    std::vector<double> hiddenLayer;
    std::vector<double> inputBias;
    std::vector<double> hiddenBias;
    std::vector<std::vector<double>> weights;


    std::vector<double> dInputBias;
    std::vector<double> dHiddenBias;
    std::vector<std::vector<double>> dWeights;

    double inputSize;
    double hiddenSize;

    NetParams(){}

    NetParams(int nInput, int nHidden){
        inputSize = nInput;
        hiddenSize = nHidden;
        inputLayer.resize(nInput);
        inputBias.resize(nInput);

        hiddenLayer.resize(nHidden);
        hiddenBias.resize(nHidden);
        weights.resize(nInput);

        dInputBias.resize(nInput);
        dHiddenBias.resize(nHidden);
        dWeights.resize(nInput);

        for(int i = 0; i < nInput; i++){
            weights[i].resize(nHidden);
            dWeights[i].resize(nHidden);
        }
    }

    void print(){
        assert(inputSize > 0);
        assert(hiddenSize > 0);
        int width = 10;
        int prec = 4;
        // std::setw(10);
        // std::setprecision(5);
        cout << "Input Layer:" << endl << "[";
        for(int i = 0; i < inputSize; i++){
            cout << setw(width) << setprecision(prec) << inputLayer[i] << ",";
        }
        cout << "]" << endl << endl;

        cout << "Input Bias:" << endl << "[";
        for(int i = 0; i < inputSize; i++){
            cout << setw(width) << setprecision(prec) << inputBias[i] << ",";
        }
        cout << "]" << endl << endl;

        cout << "Hidden Layer:" << endl << "[";
        for(int i = 0; i < hiddenSize; i++){
            cout << setw(width) << setprecision(prec) << hiddenLayer[i] << ",";
        }
        cout << "]" << endl << endl;

        cout << "Hidden Bias:" << endl << "[";
        for(int i = 0; i < hiddenSize; i++){
            cout << setw(width) << setprecision(prec) << hiddenBias[i] << ",";
        }
        cout << "]" << endl << endl;

        cout << "Weights:" << endl;
        for(int i = 0; i < inputSize; i++){
            cout << "[";
            for(int j = 0; j < hiddenSize; j++){
                cout << setw(width) << setprecision(prec) << weights[i][j] << ",";
            }
            cout << "]" << endl;
        }
    }



};
