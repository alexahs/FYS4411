#include <iostream>
#include "Hamiltonians/hamiltonian.h"
#include "random.h"
#include "simple_vmc/harmonic_osc_1d.h"
#include <fstream>
#include <iomanip>

using namespace std;

int main() {



    int n_cycles = 100000;


    run(n_cycles);



    return 0;
}
