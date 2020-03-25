#pragma once
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "Misc/particle.h"

using namespace std;

/* Just a bunch of functions that will write the desired data into
their respective folders */

void writeFileOneVariational(int, int, int, int, bool, vector<double>,
     vector<double>, vector<double>, vector<double>, vector<double>);

void printInitalSystemInfo(int, int, int, double, int);

void printFinal(int, double);

void writeFileEnergy(vector<double>&, int, int, int, string);

void writeOneBodyDensity(double**, int, string);

void writeFileAlpha(std::vector<double>& alphaVec, int numPart, int metroSteps);

void writeParticles(vector<class Particle*>, string);
