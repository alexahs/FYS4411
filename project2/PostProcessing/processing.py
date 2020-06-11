from analysis import dataAnalysisClass
from multiprocessing import Pool
import numpy as np
import shutil
import os
import re

import read_outputs
max_processes = 16


def get_analyser_data(data, test):
    '''
        Runs the dataAnalysisClass for a given set of datapoints.
    '''
    analyzer = dataAnalysisClass(data)
    analyzer.runAllAnalyses(skip = test)
    anal_dict = analyzer.returnOutput()
    return anal_dict

def get_energy_data(data):
    '''
        Calculates the mean energy of a single experiment, and returns the mean
        values and experiment parameters.

        Output map:
            vals = [sample mean, sample variance, blocking mean,
                    blocking variance, bootstrap mean, bootstrap variance,
                    particles, dimensions, hidden units, cycles, learning rate]
    '''
    means = get_analyser_data(data['Energy'], data['test'])
    vals = np.zeros(12, dtype = np.float64)
    vals[0]  = means['sample']['avg']
    vals[1]  = means['sample']['std']
    vals[2]  = means['blocking']['avg']
    vals[3]  = means['blocking']['std']
    vals[4]  = means['bootstrap']['avg']
    vals[5]  = means['bootstrap']['std']
    vals[6]  = data['particle']
    vals[7]  = data['dimensions']
    vals[8]  = data['hidden_units']
    vals[9]  = data['cycles']
    vals[10] = data['learning_rate']
    vals[11] = data['sampling']
    return vals

def split_energy_lists(results):
    '''
        Splits the list of results into compatible sublists, where each list
        only contains results with equal numbers of particles, dimensions,
        sampling methods, and cycles.
    '''

    conditions = {}
    ID = 'P{:.0f}D{:.0f}C{:.0f}S{:.0f}'
    for result in results:
        key = ID.format(result[6],result[7],result[9],result[11])
        if key not in conditions.keys():
            conditions[key] = [result]
        else:
            conditions[key].append(result)
    conditions = list(conditions.values())

    return conditions

def sort_energy_grids(results):
    '''
        Organizes a set of unordered compatible energy datapoints into a grid.
    '''
    zeros_compare = np.zeros(results.shape[0], dtype = np.float64)
    msg = 'Incompatible parameters in selected gridsearch'
    for i in [6,7,9,11]:
        assert np.array_equal(results[:,i]-results[0,i], zeros_compare), msg
    hidden_units = results[:,8]
    learning_rate = results[:,10]

    unique_HU = []
    unique_LR = []
    for i in results:
        if i[8] not in unique_HU:
            unique_HU.append(i[8])
        if i[10] not in unique_LR:
            unique_LR.append(i[10])

    unique_HU = np.sort(np.array(unique_HU, np.float64))
    unique_LR = np.sort(np.array(unique_LR, np.float64))

    LR, HU = np.meshgrid(unique_LR, unique_HU)
    energy_grid_sample = np.zeros((len(unique_HU), len(unique_LR)))
    energy_grid_blocking = np.zeros((len(unique_HU), len(unique_LR)))
    energy_grid_bootstrap = np.zeros((len(unique_HU), len(unique_LR)))
    err_grid_sample = np.zeros((len(unique_HU), len(unique_LR)))
    err_grid_blocking = np.zeros((len(unique_HU), len(unique_LR)))
    err_grid_bootstrap = np.zeros((len(unique_HU), len(unique_LR)))
    msg2 = 'Multiple energies found for same conditions.'
    for i,(a,b) in enumerate(zip(LR, HU)):
        for j,(c,d) in enumerate(zip(a, b)):
            cond_1 = results[:,10] == c
            cond_2 = results[:,8] == d
            idx = np.where(np.logical_and(cond_1, cond_2))
            assert len(idx) == 1, msg2
            idx = idx[0]
            energy_grid_sample[i,j] = results[idx].squeeze()[0]
            err_grid_sample[i,j] = results[idx].squeeze()[1]
            energy_grid_blocking[i,j] = results[idx].squeeze()[2]
            err_grid_blocking[i,j] = results[idx].squeeze()[3]
            energy_grid_bootstrap[i,j] = results[idx].squeeze()[4]
            err_grid_bootstrap[i,j] = results[idx].squeeze()[5]

    return energy_grid_sample, err_grid_sample, energy_grid_blocking, err_grid_blocking, energy_grid_bootstrap, err_grid_bootstrap, LR, HU

def get_energy_grids(data, test = False):
    '''
        Calculates the mean energies after optimization is complete for all
        points given in a list of data dicts.
    '''
    pool = Pool(processes = max_processes)
    for i in range(len(data)):
        data[i]['test'] = test
    results = np.zeros((len(data), 12), dtype = np.float64)
    count = 0
    print(f'Loading {0:.2%}', end ='')
    for n,i in enumerate(pool.imap_unordered(get_energy_data, data)):
        count += 1
        print(f'\rLoading {count/len(data):.2%}', end ='')
        results[n] = i
    print()
    results = split_energy_lists(results)
    sorted_results = {}
    for result in results:
        key = f'P{result[0][6]:.0f}D{result[0][7]:.0f}C{result[0][9]:.0f}S{result[0][11]:.0f}'
        sorted_results[key] = {}
        energy_grid_sample, err_grid_sample, energy_grid_blocking, err_grid_blocking, energy_grid_bootstrap, err_grid_bootstrap, LR, HU = \
        sort_energy_grids(np.array(result))
        sorted_results[key]['sample'] = energy_grid_sample
        sorted_results[key]['sample_err'] = err_grid_sample
        sorted_results[key]['blocking'] = energy_grid_blocking
        sorted_results[key]['blocking_err'] = err_grid_blocking
        sorted_results[key]['bootstrap'] = energy_grid_bootstrap
        sorted_results[key]['bootstrap_err'] = err_grid_bootstrap
        sorted_results[key]['LR'] = LR
        sorted_results[key]['HU'] = HU
    return sorted_results

def save_energy_grids(sorted_results, savename, autodelete = False):
    '''
        Saves the energy grids to file
    '''
    savename = f'../DataProcessed/{savename}'

    if not os.path.exists('../DataProcessed/'):
        os.mkdir('../DataProcessed/')

    if os.path.isdir(savename):
        if autodelete:
            shutil.rmtree(savename)
        else:
            while True:
                delete = input('Delete Previously Saved Data? (y/n)')
                if delete == 'y':
                    shutil.rmtree(savename)
                    break
                elif delete == 'n':
                    exit()
                else:
                    print('Invalid input: {delete}, try again.')

    os.mkdir(savename)
    for key, value in sorted_results.items():
        name = f'{savename}/arr_{key}_{{}}'
        np.save(name.format('sample'), value['sample'])
        np.save(name.format('blocking'), value['blocking'])
        np.save(name.format('bootstrap'), value['bootstrap'])
        np.save(name.format('LR'), value['LR'])
        np.save(name.format('HU'), value['HU'])

        np.save(name.format('sample_err'), value['sample_err'])
        np.save(name.format('blocking_err'), value['blocking_err'])
        np.save(name.format('bootstrap_err'), value['bootstrap_err'])

def load_energy_grids(savename):
    '''
        Loads the energy grids from file
    '''
    savename = f'../DataProcessed/{savename}'
    files = os.listdir(savename)
    sorted_results = {}
    for f in files:
        pattern = r'arr_(P\d+D\d+C\d+S\d+)_(.*)\.npy'
        key1, key2  = re.findall(pattern, f)[0]
        if key1 not in sorted_results.keys():
            sorted_results[key1] = {}
        sorted_results[key1][key2] = np.load(savename + '/' + f)
    return sorted_results

def split_position_lists(data):
    '''
        Splits the list of results into compatible sublists, where each list
        only contains results with equal numbers of particles, dimensions,
        and cycles.
    '''

    conditions = {}
    ID = 'P{:.0f}D{:.0f}C{:.0f}S{:.0f}'
    for d in data:
        key = ID.format(d['particle'], d['dimensions'], d['cycles'], d['sampling'])
        if key not in conditions.keys():
            conditions[key] = [d]
        else:
            conditions[key].append(d)
    conditions = list(conditions.values())
    return conditions

def sort_position_grids(results):
    '''
        Organizes a set of unordered compatible position datapoints into a grid.
    '''
    unique_HU = []
    unique_LR = []
    for i in results:
        if i['hidden_units'] not in unique_HU:
            unique_HU.append(i['hidden_units'])
        if i['learning_rate'] not in unique_LR:
            unique_LR.append(i['learning_rate'])

    unique_HU = np.sort(np.array(unique_HU, np.float64))
    unique_LR = np.sort(np.array(unique_LR, np.float64))

    LR, HU = np.meshgrid(unique_LR, unique_HU)
    pos_grid = [[0 for i in unique_HU] for j in unique_LR]
    for i,(a,b) in enumerate(zip(LR, HU)):
        for j,(c,d) in enumerate(zip(a, b)):
            for result in results:
                if result['learning_rate'] == c and result['hidden_units'] == d:
                    pos_grid[j][i] = result['pos']
                    break
    return pos_grid, LR, HU

def get_position_grids(data):
    '''
        Calculates the mean positions after optimization is complete for all
        points given in a list of data dicts.
    '''
    results = split_position_lists(data)
    sorted_results = {}
    for result in results:
        key = f'P{result[0]["particle"]:.0f}D{result[0]["dimensions"]:.0f}C{result[0]["cycles"]:.0f}S{result[0]["sampling"]:.0f}'
        sorted_results[key] = {}
        pos_grid, LR, HU = sort_position_grids(result)
        sorted_results[key]['pos'] = pos_grid
        sorted_results[key]['LR'] = LR
        sorted_results[key]['HU'] = HU
    return sorted_results

if __name__ == '__main__':
    energies = read_outputs.read_energy_samples()
    positions = read_outputs.read_pos_samples()
    sorted_positions = get_position_grids(positions)
    sorted_energies = get_energy_grids(energies, test = False)

    save_energy_grids(sorted_energies, 'run', autodelete = True)
    loaded_energies = load_energy_grids('run')

    for k,v in sorted_energies.items():
        assert np.array_equal(v['sample'], loaded_energies[k]['sample'])
        assert np.array_equal(v['blocking'], loaded_energies[k]['blocking'])
        assert np.array_equal(v['bootstrap'], loaded_energies[k]['bootstrap'])
        assert np.array_equal(v['sample_err'], loaded_energies[k]['sample_err'])
        assert np.array_equal(v['blocking_err'], loaded_energies[k]['blocking_err'])
        assert np.array_equal(v['bootstrap_err'], loaded_energies[k]['bootstrap_err'])
        assert np.array_equal(v['LR'], loaded_energies[k]['LR'])
        assert np.array_equal(v['HU'], loaded_energies[k]['HU'])
