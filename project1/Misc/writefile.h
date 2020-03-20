#pragma once
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>

using namespace std;

void writeFileOneVariational(bool, int, int, int, int, bool, vector<double>,
     vector<double>, vector<double>, vector<double>, vector<double>, double, double);

void printInitalSystemInfo(int, int, int, double, int);

void printFinal(int, double);

void writeFileEnergy(std::vector<double>& energySamples,
                     int numDim,
                     int numPart,
                     int metroSteps,
                     double alpha,
                     double timeStep);
