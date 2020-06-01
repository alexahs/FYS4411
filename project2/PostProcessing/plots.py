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

def plot_energies(grids, dimensions = 2, expected = None):
    '''
        Plots energies using energy data from processing.py

        If dimensions is 2, makes a 2D contour plot
        If dimensions is 3, makes a 3D surface plot

        Expected should be the expected energy value (float)
    '''
    msg = '$\Delta E={:g}$ at $\eta={:g}$, $H={:g}$'
    for key, value in grids.items():
        means = ['sample', 'bootstrap', 'blocking']
        if dimensions == 2:
            for mean_type in means:
                Z = value[mean_type]
                if expected is not None:
                    Z = np.abs(Z - expected)
                    idx = np.unravel_index(np.argmin(Z), Z.shape)
                    p1 = value['LR'][idx[0],idx[1]]
                    p2 = value['HU'][idx[0],idx[1]]
                    Em = Z[idx[0],idx[1]]
                    Z = np.log(Z)
                else:
                    cutoffs = [0.49, 0.51]
                    Z[Z < cutoffs[0]] = cutoffs[0]
                    Z[Z > cutoffs[1]] = cutoffs[1]
                plt.figure()
                plt.contourf(value['LR'], value['HU'], Z, 400, cmap='magma')
                plt.xlabel('Learning Rate $\eta$')
                plt.ylabel('Number of Nodes $H$')
                cbar = plt.colorbar()
                plt.axis(aspect='image')
                if expected is not None:
                    plt.plot(p1, p2, 'xr', ms = 10,  label = msg.format(Em, p1, p2))
                    plt.legend()
                    msg2 = f'Log of |{mean_type.capitalize()} Energy - Expected Energy|'
                    cbar.set_label(msg2)
                else:
                    cbar.set_label(f'Mean {mean_type.capitalize()} Energy')
            plt.show()
        elif dimensions == 3:
            for mean_type in means:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                Z = value[mean_type]
                if expected is not None:
                    Z = np.abs(Z - expected)
                    idx = np.unravel_index(np.argmin(Z), Z.shape)
                    p1 = value['LR'][idx[0],idx[1]]
                    p2 = value['HU'][idx[0],idx[1]]
                    Em = Z[idx[0],idx[1]]
                    Z = np.log(Z)
                    p3 = Z[idx[0],idx[1]]
                else:
                    cutoffs = [0.47, 0.53]
                    Z[Z < cutoffs[0]] = cutoffs[0]
                    Z[Z > cutoffs[1]] = cutoffs[1]
                ax.plot_surface(value['LR'], value['HU'], Z, cmap = 'magma')
                ax.set_xlabel('Learning Rate $\eta$')
                ax.set_ylabel('Number of Nodes')
                if expected is not None:
                    ax.scatter(p1, p2, p3, label = msg.format(Em, p1, p2))
                    plt.legend()
                    msg2 = f'Log of |{mean_type.capitalize()} Energy - Expected Energy|'
                    ax.set_zlabel(msg2)
                else:
                    ax.set_zlabel(f'Mean {mean_type.capitalize()} Energy')
            plt.show()

if __name__ == '__main__':
    # optimizations = read_outputs.read_optimization()
    # energies = read_outputs.read_energy_samples()
    #
    grids = processing.load_energy_grids('run_1')
    plot_energies(grids, 3, 0.5)
