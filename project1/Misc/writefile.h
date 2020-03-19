#pragma once
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "Misc/particle.h"

using namespace std;

void writeFileOneVariational(int, int, int, int, bool, vector<double>,
     vector<double>, vector<double>, vector<double>, vector<double>);

void printInitalSystemInfo(int, int, int, double, int);

void printFinal(int, double);

void writeFileEnergy(vector<double>&, int, int, int, string);

void writeParticles(vector<class Particle*>);
