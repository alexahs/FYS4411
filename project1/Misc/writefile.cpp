#include "writefile.h"

void writeFileOneVariational(int numDim, int numPart, int metroSteps, int equilSteps,
    vector<double> alpha, vector<double> energy, vector<double> energy2,
    vector<double> variance, vector<double> acceptRatio, vector<double> sumRiSquared)
{
    std::string filename = "./Data/vmc_";
    filename.append(to_string(numDim) + "d_");
    filename.append(to_string(numPart) + "p.csv");

    ofstream outfile;
    outfile.open(filename, ofstream::out | ofstream::trunc);
    // write header for columns
    outfile << "Alpha,Energy,Energy2,Variance,AcceptRatio,SumRiSquared" << endl;
    for (int i=0; i<alpha.size(); i++) {
        outfile << alpha[i] << ",";
        outfile << energy[i] << ",";
        outfile << energy2[i] << ",";
        outfile << variance[i] << ",";
        outfile << acceptRatio[i] << ",";
        outfile << sumRiSquared[i] << endl;
    }
    outfile.close();
}
