#include "writefile.h"

void writeFileOneVariational(int numDim, int numPart, int metroSteps, int equilSteps,
    bool numericalDoubleDerviative,
    vector<double> alpha, vector<double> energy, vector<double> energy2,
    vector<double> variance, vector<double> acceptRatio)
{
    std::string filename = "./Data/vmc_";
    filename.append(to_string(numDim) + "d_");
    filename.append(to_string(numPart) + "p_");
    if (numericalDoubleDerviative) {
        filename.append("num.csv");
    } else {
        filename.append("ana.csv");
    }

    ofstream outfile;
    outfile.open(filename, ofstream::out | ofstream::trunc);
    // write header for columns
    outfile << "Alpha,Energy,Energy2,Variance,AcceptRatio" << endl;
    for (int i=0; i<alpha.size(); i++) {
        outfile << alpha[i] << ",";
        outfile << energy[i] << ",";
        outfile << energy2[i] << ",";
        outfile << variance[i] << ",";
        outfile << acceptRatio[i] << endl;
    }
    outfile.close();
}
