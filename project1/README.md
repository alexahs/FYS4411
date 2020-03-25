# Variational Monte Carlo

### Abstract

*In the research presented, we study Bose-Einstein condensates of trapped bosons in an elliptical harmonic oscillator trap, by numerical approaches. We represent the systems with trial wave functions with two variational parameters. The aim is to identify and study the ground states of the systems, by performing variational Monte Carlo methods, namely the Metropolis algorithm and the Metropolis-Hastings algorithm. Lastly, we study the correlation factor in the trial wave function, by evaluating the one-body density with and without the Jastrow factor. For spherical Gaussian wave functions without correlation, we find excellent agreement with the theoretical ground state energies. We also show that the correlation, governed by the Jastrow factor, leads to a more spread out system.*

### Folder Structure

The following folders contains base classes and derived classes:

* Hamiltonian
    * hamiltonian.cpp (Base class)
        * EllipticHarmonicOscillator (Derived class)
        * HarmonicOscillator (Derived class)
* InitialStates
    * InitialState (Base class)
        * RandomUniform (Derived class)
        * UniformLattice (Derived class)
* WaveFunction
    * WaveFunction (Base class)
        * SimpleGaussian (Derived class)
        * Correlated (Derived class)

The folder Misc contains miscellaneous classes that are used in the code

* Misc
    * Particle
    * System, Superclass which includes:
        * Particles (vector of particle)
        * One Hamiltonian class
        * One InitialState class
        * One WaveFunction class
    * Sampler
    * WfSampler

The Math class contains the RNG used

* Math
    * Random (class)

### Compile and Run

The main configurations are done in main.cpp! The number of particles is passed
as a command line argument, and all the other variables must be tweaked inside main.cpp.

**compile**
```
$ ./compile_project
```
The dependencies are mainly the GNU compiler gcc/g++ and cmake. Optionally OpenMP, and if you don't want to parallelize, just comment the #pragma lines for parallelization in main.cpp.

**run**
Comment/uncomment the analyses you want to run inside main.cpp, then
```
$ ./vmc <numberOfParticles>
```
Example:
```
$ ./vmc 10
```
The obtained data will be stored in the Data directory (ignored by git).

**analysis**
This is done with python files (./Post Analysis/plot.py etc.). Comment/uncomment the desired post analysis function at the bottom of plot.py and run
```
$ python plot.py <dimensions> <particles> <log2steps>
```
Example:
```
$ python plot.py 3 10 20
```
