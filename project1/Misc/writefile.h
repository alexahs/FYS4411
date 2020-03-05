#pragma once
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>

using namespace std;

void writeFileOneVariational(int, int, int, int, bool, vector<double>,
     vector<double>, vector<double>, vector<double>, vector<double>);

void printInitalSystemInfo(int, int, int, double, int);

void printFinal(int, double);
