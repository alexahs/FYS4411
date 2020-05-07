# FYS4411 Project 2:
## The restricted Boltzmann machine applied to the quantum manybody problem

Compile by running the shell script `compile_project.`, creates an executable `rbm`.



## Progress
Main parts of the program are implemented, but it is most likely full of bugs...
With small enough initialized positions and no interaction, it produces the analytical
energy instantly. With a bigger spread of initial positions, the energy goes negative, and the
optimizer has a hard time optimizing the parameters.

## TODO:
* Test the code to be sure that it is running correctly (test functions?)
* Implement importance sampling
* Implement a better optimizer (only have standard gradient descent so far)
* Implement Gibbs sampling
