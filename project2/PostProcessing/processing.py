from analysis import dataAnalysisClass
from multiprocessing import Pool
import numpy as np
import shutil
import sys
import os
import re

import read_outputs
max_processes = 8

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
    vals = np.zeros(10, dtype = np.float64)
    vals[0]  = means['sample']['avg']
    vals[1]  = means['sample']['std']
    vals[2]  = means['blocking']['avg']
    vals[3]  = means['blocking']['std']

    vals[4]  = data['particle']
    vals[5]  = data['dimensions']
    vals[6]  = data['hidden_units']
    vals[7]  = data['cycles']
    vals[8] = data['learning_rate']
    vals[9] = data['sampling']
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
        key = ID.format(result[4],result[5],result[7],result[9])
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
    for i in [4,5,7,9]:
        assert np.array_equal(results[:,i]-results[0,i], zeros_compare), msg
    hidden_units = results[:,6]
    learning_rate = results[:,8]

    unique_HU = []
    unique_LR = []
    for i in results:
        if i[6] not in unique_HU:
            unique_HU.append(i[6])
        if i[8] not in unique_LR:
            unique_LR.append(i[8])

    unique_HU = np.sort(np.array(unique_HU, np.float64))
    unique_LR = np.sort(np.array(unique_LR, np.float64))

    LR, HU = np.meshgrid(unique_LR, unique_HU)
    energy_grid_sample = np.zeros((len(unique_HU), len(unique_LR)))
    energy_grid_blocking = np.zeros((len(unique_HU), len(unique_LR)))
    err_grid_sample = np.zeros((len(unique_HU), len(unique_LR)))
    err_grid_blocking = np.zeros((len(unique_HU), len(unique_LR)))
    msg2 = 'Multiple energies found for same conditions.'
    for i,(a,b) in enumerate(zip(LR, HU)):
        for j,(c,d) in enumerate(zip(a, b)):
            cond_1 = results[:,8] == c
            cond_2 = results[:,6] == d
            idx = np.where(np.logical_and(cond_1, cond_2))
            assert len(idx) == 1, msg2
            idx = idx[0]
            energy_grid_sample[i,j] = results[idx].squeeze()[0]
            err_grid_sample[i,j] = results[idx].squeeze()[1]
            energy_grid_blocking[i,j] = results[idx].squeeze()[2]
            err_grid_blocking[i,j] = results[idx].squeeze()[3]

    return energy_grid_sample, err_grid_sample, energy_grid_blocking, err_grid_blocking, LR, HU

def save_energy_grids(data, savename, test = False):
    '''
        Calculates the mean energies after optimization is complete for all
        points given in a list of data dicts.
    '''

    savename = f'../DataProcessed/{savename}'

    if len(sys.argv) > 1 and sys.argv[1] == '-o':
        if os.path.isdir(savename):
            shutil.rmtree(savename)
        os.mkdir(savename)
    elif os.path.isdir(savename):
        while True:
            delete = input('Delete Previously Saved Data? (y/n)')
            if delete == 'y':
                shutil.rmtree(savename)
                os.mkdir(savename)
                break
            elif delete == 'n':
                exit()
            else:
                print('Invalid input: {delete}, try again.')
    else:
        os.mkdir(savename)

    pool = Pool(processes = max_processes)
    for i in range(len(data)):
        data[i]['test'] = test
    results = np.zeros((len(data), 10), dtype = np.float64)
    count = 0
    print(f'Loading {0:.2%}', end ='')
    for n,i in enumerate(pool.imap_unordered(get_energy_data, data)):
        count += 1
        print(f'\rLoading {count/len(data):.2%}', end ='')
        results[n] = i
    print()
    results = split_energy_lists(results)
    for result in results:
        key = f'P{result[0][4]:.0f}D{result[0][5]:.0f}C{result[0][7]:.0f}S{result[0][9]:.0f}'

        energy_grid_sample, err_grid_sample, energy_grid_blocking, err_grid_blocking, LR, HU = \
        sort_energy_grids(np.array(result))

        name = f'{savename}/arr_{key}_{{}}'
        np.save(name.format('sample'), energy_grid_sample)
        np.save(name.format('blocking'), energy_grid_blocking)
        np.save(name.format('LR'), LR)
        np.save(name.format('HU'), HU)
        np.save(name.format('blocking_err'), err_grid_blocking)

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

def get_best_energy_data(grid, expected):
    '''
        Returns the best HU and LR, as well as their indices, for an energy
        grid, given an expected value.
    '''
    diffs = np.abs(grid['sample']-expected)
    idx = np.unravel_index(diffs.argmin(), diffs.shape)
    HU = grid['HU'][idx]
    LR = grid['LR'][idx]
    E = grid['blocking'][idx]
    dE = diffs[idx]
    data = {'HU':HU, 'LR':LR, 'E':E, 'dE':dE, 'idx':idx}
    return data

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == '-o':
        if os.path.exists('../DataProcessed/'):
            shutil.rmtree('../DataProcessed/')
        os.mkdir('../DataProcessed/')
    elif not os.path.exists('../DataProcessed/'):
        os.mkdir('../DataProcessed/')
    elif os.listdir('../DataProcessed/'):
        while True:
            delete = input('Delete Previously Processed Data? (y/n)')
            if delete == 'y':
                shutil.rmtree('../DataProcessed/')
                os.mkdir('../DataProcessed/')
                break
            elif delete == 'n':
                break
            else:
                print('Invalid input: {delete}, try again.')

    energies = read_outputs.read_energy_samples()
    save_energy_grids(energies, 'run')
