from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

import read_outputs
import processing

def plot_optimizations(data):
    '''
        Plots the energy and variance of the calculated energies during the
        optimization process.
    '''
    E = data['Energy']
    var = data['Variance']
    steps = np.arange(1, len(E)+1)

    plt.plot(steps, E, label = 'Energy')
    alpha = 0.1
    plt.plot(steps, E+var/2, 'r--', alpha = alpha)
    plt.plot(steps, E-var/2, 'r--', alpha = alpha)
    plt.fill_between(steps, E-var/2, E+var/2, color = 'r', alpha = alpha,
                     label = 'Variance')
    plt.xlim(np.min(steps), np.max(steps))
    plt.ylim(np.min([0, np.min(E-var/2)]), np.max(E+var/2))
    plt.grid()
    plt.xlabel('Optimization Step')
    plt.ylabel('Energy $[eV]$ and Variance $[eV]$')
    plt.legend()
    plt.show()

def plot_energies(data, test = False):
    '''
        Plots energies using energy data from processing.py
    '''
    grids = processing.get_energy_grids(data, test)

    for key, value in grids.items():

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        Z = value['sample']
        cutoffs = [0.48, 0.53]
        Z[Z < cutoffs[0]] = cutoffs[0]
        Z[Z > cutoffs[1]] = cutoffs[1]

        ax.plot_surface(value['LR'], value['HU'], Z, cmap = 'magma')
        ax.set_title('Surface plot')
    plt.show()

if __name__ == '__main__':
    optimizations = read_outputs.read_optimization()
    energies = read_outputs.read_energy_samples()

    plot_energies(energies, test = True)
