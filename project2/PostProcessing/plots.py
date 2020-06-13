from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import shutil
import os

import read_outputs
import processing

def plot_optimizations(data, key, ext = '.pdf'):
    '''
        Plots the energy and variance of the calculated energies during the
        optimization process.
    '''

    data = data[key]

    if not os.path.exists('../Plots/'):
        os.mkdir('../Plots/')

    save_dir = '../Plots/Convergence/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    E = data['Energy']
    var = data['Variance']
    steps = np.arange(1, len(E)+1)

    plt.figure()
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
    plt.savefig(save_dir + key + ext)
    plt.close()

def plot_energies(grids, key, dimensions = 2, expected = None, ext = '.pdf'):
    '''
        Plots energies using energy data from processing.py

        If dimensions is 2, makes a 2D contour plot
        If dimensions is 3, makes a 3D surface plot

        Expected should be the expected energy value (float)
    '''

    grid = grids[key]

    if not os.path.exists('../Plots/'):
        os.mkdir('../Plots/')

    save_dir = '../Plots/Energies/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    msg = '$\Delta E={:g}$ at $\eta={:g}$, $H={:g}$'
    means = ['sample', 'bootstrap', 'blocking']
    if dimensions == 2:
        for mean_type in means:
            Z = grid[mean_type]
            if expected is not None:
                Z = np.abs(Z - expected)
                idx = np.unravel_index(np.argmin(Z), Z.shape)
                p1 = grid['LR'][idx[0],idx[1]]
                p2 = grid['HU'][idx[0],idx[1]]
                Em = Z[idx[0],idx[1]]
                Z = np.log(Z)
            else:
                cutoffs = [0, 4]
                Z[Z < cutoffs[0]] = cutoffs[0]
                Z[Z > cutoffs[1]] = cutoffs[1]
            plt.figure()
            plt.pcolormesh(grid['LR'], grid['HU'], Z, cmap='magma')
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
            plt.savefig(save_dir + key + f'_{mean_type}{ext}')
            plt.close()
    elif dimensions == 3:
        for mean_type in means:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            Z = grid[mean_type]
            if expected is not None:
                Z = np.abs(Z - expected)
                idx = np.unravel_index(np.argmin(Z), Z.shape)
                p1 = grid['LR'][idx[0],idx[1]]
                p2 = grid['HU'][idx[0],idx[1]]
                Em = Z[idx[0],idx[1]]
                Z = np.log(Z)
                p3 = Z[idx[0],idx[1]]
            else:
                cutoffs = [0, 2]
                Z[Z < cutoffs[0]] = cutoffs[0]
                Z[Z > cutoffs[1]] = cutoffs[1]
            ax.plot_surface(grid['LR'], grid['HU'], Z, cmap = 'magma')
            ax.set_xlabel('Learning Rate $\eta$')
            ax.set_ylabel('Number of Nodes')
            if expected is not None:
                ax.scatter(p1, p2, p3, label = msg.format(Em, p1, p2))
                plt.legend()
                msg2 = f'Log of |{mean_type.capitalize()} Energy - Expected Energy|'
                ax.set_zlabel(msg2)
            else:
                ax.set_zlabel(f'Mean {mean_type.capitalize()} Energy')
            plt.savefig(save_dir + key + f'_{mean_type}{ext}')
            plt.close()

def plot_err(grids, key, dimensions = 2, ext = '.pdf'):
    '''
        Plots energies using energy data from processing.py

        If dimensions is 2, makes a 2D contour plot
        If dimensions is 3, makes a 3D surface plot

        Expected should be the expected energy value (float)
    '''

    grid = grids[key]

    if not os.path.exists('../Plots/'):
        os.mkdir('../Plots/')

    save_dir = '../Plots/Errors/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    msg = '$\Delta E={:g}$ at $\eta={:g}$, $H={:g}$'
    means = ['sample_err', 'bootstrap_err', 'blocking_err']
    if dimensions == 2:
        for mean_type in means:
            Z = grid[mean_type]
            cutoffs = [0, 4]
            Z[Z < cutoffs[0]] = cutoffs[0]
            Z[Z > cutoffs[1]] = cutoffs[1]
            plt.figure()
            plt.pcolormesh(grid['LR'], grid['HU'], Z, cmap='magma')
            plt.xlabel('Learning Rate $\eta$')
            plt.ylabel('Number of Nodes $H$')
            cbar = plt.colorbar()
            plt.axis(aspect='image')
            label = (mean_type.split('_'))[0].capitalize()
            cbar.set_label(f'Std. Error of {label} Energy')
            plt.savefig(save_dir + key + f'_{mean_type}{ext}')
            plt.close()
    elif dimensions == 3:
        for mean_type in means:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            Z = grid[mean_type]
            cutoffs = [0, 2]
            Z[Z < cutoffs[0]] = cutoffs[0]
            Z[Z > cutoffs[1]] = cutoffs[1]
            label = (mean_type.split('_'))[0].capitalize()
            ax.plot_surface(grid['LR'], grid['HU'], Z, cmap = 'magma')
            ax.set_xlabel('Learning Rate $\eta$')
            ax.set_ylabel('Number of Nodes')
            ax.set_zlabel(f'Std. Error of  {label} Energy')
            plt.savefig(save_dir + key + f'_{mean_type}{ext}')
            plt.close()

def plot_pos(grids, key, i, j, ext = '.pdf'):
    '''
        Plots the position density at a given point
    '''

    grid = grids[key]['pos'][i][j]
    HU = grids[key]['HU'][i][j]
    LR = grids[key]['LR'][i][j]

    if not os.path.exists('../Plots/'):
        os.mkdir('../Plots/')

    save_dir = '../Plots/Positions/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    grid = np.linalg.norm(grid, axis = 1)
    maxwell = stats.maxwell

    x = np.linspace(np.min(grid), np.max(grid), 1000)

    dist_label = 'Boltzmann Distribution Best Fit'
    dist_params = {
                   'label'      : dist_label,
                   'color'      : 'r'
                  }

    y = maxwell.pdf(x, *maxwell.fit(grid))
    plt.figure()
    plt.plot(x, y, **dist_params)

    hist_params = {
                    'density'   : True,
                    'stacked'   : True,
                    'bins'      : 200,  #grid.shape[0]//50,
                    'label'     : 'Experimental Results',
                    'color'     : 'C0'
                  }
    units = r'$\left[ \sqrt{\frac{\hbar}{m \omega_{ho}}} \ \right]$'
    plt.hist(grid, **hist_params)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('Probability Density')
    plt.xlabel('Distance from Origin ' + units)
    plt.xlim([np.min(grid), np.max(grid)])
    plt.grid()
    plt.legend()

    plt.savefig(save_dir + key + f'HU{HU}LR{LR}' + ext)
    plt.close()

def plot_sigmas(data, key, expected, ext = '.pdf'):
    '''
        Plots the energy deviation from expected as functions of sigma and
        sigma_init.
    '''
    data = data[key]

    if not os.path.exists('../Plots/'):
        os.mkdir('../Plots/')

    save_dir = '../Plots/Sigmas/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    idx = np.isfinite(data['sigmas_E'])

    sigmas = data['sigmas'][idx]
    energies = data['sigmas_E'][idx]

    diffs = np.abs(energies-expected)
    best_E = np.min(diffs)
    best_sigma = sigmas[np.argmin(diffs)]

    label1 = 'Energies'
    label2 = f'Best $\sigma = {best_sigma:g}$'
    plt.figure()
    plt.grid()
    plt.semilogy([best_sigma], [best_E], 'rx', ms = 15, label = label2)
    plt.semilogy(sigmas, diffs, label = label1)
    plt.xlabel('$\sigma$')
    plt.ylabel('Energy Deviance from Expected Value')
    plt.legend()
    plt.xlim(np.min(sigmas), np.max(sigmas))
    plt.savefig(save_dir + key + f'_sigma{ext}')
    plt.close()


    # plt.figure()
    # plt.grid()
    # idx = np.isfinite(data['sigma_inits_E'])
    #
    # sigmas = data['sigma_inits'][idx]
    # energies = data['sigma_inits_E'][idx]
    #
    # diffs = np.abs(energies-expected)
    # best_E = np.min(diffs)
    # best_sigma = sigmas[np.argmin(diffs)]
    #
    # label1 = 'Energies'
    # label2 = f'Best $\sigma_{{init}} = {best_sigma:g}$'
    # plt.semilogy([best_sigma], [best_E], 'rx', ms = 15, label = label2)
    # plt.semilogy(sigmas, diffs, label = label1)
    # plt.xlabel('$\sigma_{{init}}$')
    # plt.ylabel('Energy Deviance from Expected Value')
    # plt.legend()
    # plt.xlim(np.min(sigmas), np.max(sigmas))
    #
    # plt.savefig(save_dir + key + f'_sigma_init{ext}')
    # plt.close()

def get_energy_table(data):
    '''
        Creates a LaTEX formatted table including the errors, and specific
        hyperparameters for the best-fitting energies
    '''
    # for key, value in data.items():
        # pattern =

if __name__ == '__main__':

    # iters = {
    #          'optim': 131072,
    #          'main' : 1048576,
    #          'sigma': 1024
    #         }

    iters = {
             'optim': 8192,
             'main' : 65536,
             'sigma': 256
            }

    if not os.path.exists('../Plots/'):
        os.mkdir('../Plots/')
    elif os.listdir('../Plots/'):
        while True:
            delete = input('Delete Previously Saved Data? (y/n)')
            if delete == 'y':
                shutil.rmtree('../Plots/')
                os.mkdir('../Plots/')
                break
            elif delete == 'n':
                break
            else:
                print('Invalid input: {delete}, try again.')

    optimizations = read_outputs.read_optimization()
    grids = processing.load_energy_grids('run')
    positions = read_outputs.read_pos_samples()
    sorted_positions = processing.get_position_grids(positions)
    sigmas = read_outputs.read_optimized_sigmas()
    ext = '.png'

    optim_keys = ['P1D1C{:d}S1', 'P1D1C{:d}S2', 'P1D1C{:d}S3']
    for n,i in enumerate(optim_keys):
        optim_keys[n] = i.format(iters['optim'])
    for i in optim_keys:
        plot_optimizations(optimizations, i, ext = ext)

    energy_err_keys = ['P1D1C{:d}S1', 'P1D1C{:d}S2', 'P1D1C{:d}S3']
    for n,i in enumerate(energy_err_keys):
        energy_err_keys[n] = i.format(iters['main'])
    for i in energy_err_keys:
        plot_energies(grids, i, 2, ext = ext)
        plot_err(grids, i, 2, ext = ext)

    position_keys = ['P1D1C{:d}S1', 'P1D1C{:d}S2', 'P1D1C{:d}S3', 'P2D2C{:d}S2']
    for n,i in enumerate(position_keys):
        position_keys[n] = i.format(iters['main'])
    expected = [0.5, 0.5, 0.5, 3]
    for i,j in zip(position_keys, expected):
        data = processing.get_best_energy_data(grids[i], j)
        plot_pos(sorted_positions, i, *data['idx'], ext = ext)

    sigma_keys = ['P1D1C{:d}S3', 'P2D2C{:d}S3']
    expected = [0.5, 3]
    for n,i in enumerate(sigma_keys):
        sigma_keys[n] = i.format(iters['sigma'])
    for i,j in zip(sigma_keys, expected):
        plot_sigmas(sigmas, i, 3, ext = ext)
