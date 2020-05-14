# FYS4411 Project 2:
## The restricted Boltzmann machine applied to the quantum manybody problem

Compile by running `./compile_project`, creates the executable `rbm`.



## Progress
Main parts of the program are implemented, not thoroughly tested, but produces somewhat correct results.
Still using a naive optimizer, not sure how well it is working, since the randomly initialized weights
seem to produce results which are already close to the energy minimum.

## TODO:
* Test the code to be sure that it is running correctly.
* Implement a better optimizer (only have standard gradient descent so far)
* Implement Gibbs sampling
