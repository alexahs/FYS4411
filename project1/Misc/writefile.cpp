#include "writefile.h"


void writeFileEnergy(std::vector<double>& energySamples, int numDim, int numPart, int metroSteps){
    std::string filename = "./Data/vmc_energysamples_";
    filename.append(to_string(numDim) + "d_");
    filename.append(to_string(numPart) + "p_");
    filename.append("2pow" + to_string(int(log2(metroSteps))) + "steps.bin");

    ofstream outfile;
    outfile.open(filename, ios::out | ios::binary | ios::trunc);
    outfile.write(reinterpret_cast<const char*> (energySamples.data()),energySamples.size()*sizeof(double));
    outfile.close();

    cout << " * Results written to " << filename << endl;


}


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

    cout << " * Results written to " << filename << endl;
}

void printInitalSystemInfo(int numberOfDimensions, int numberOfParticles,
        int numberOfSteps, double equilibration, int numberOfParameters) {
    cout << endl;
    cout << " -------- System info -------- " << endl;
    cout << " * Number of dimensions : " << numberOfDimensions << endl;
    cout << " * Number of particles  : " << numberOfParticles << endl;
    cout << " * Number of Metropolis steps run : " << numberOfSteps << endl;
    cout << " * Number of equilibration steps  : " << int(numberOfSteps*equilibration) << endl;
    cout << " * Number of parameters : " << numberOfParameters << endl << endl;

    int numOfColumns = 4;
    for (int i=1; i<=numberOfParameters; i++) {
        cout << " Parameter " << i << " ";
        numOfColumns ++;
    }
    cout << "| Energy     ";
    cout << "| Energy^2   ";
    cout << "| Variance   ";
    cout << "| AcceptRatio" << endl << " ";

    for (int i=1; i<numOfColumns; i++) {
        cout << "------------+";
    }
    cout << "------------" << endl << " ";
}

void printFinal(int numberOfParameters, double elapsedTime) {
    int numOfColumns = 4 + numberOfParameters;
    for (int i=1; i<numOfColumns; i++) {
        cout << "-------------";
    }
    cout << "------------" << endl;
    cout << " * Execution time: " << elapsedTime << " ms" << endl << endl;
}
