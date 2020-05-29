from analysis import dataAnalysisClass
from multiprocessing import Pool
import numpy as np

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
    vals = np.zeros(11, dtype = np.float64)
    vals[0]  = means['sample']['avg']
    vals[1]  = means['sample']['var']
    vals[2]  = means['blocking']['avg']
    vals[3]  = means['blocking']['var']
    vals[4]  = means['bootstrap']['avg']
    vals[5]  = means['bootstrap']['var']
    vals[6]  = data['particle']
    vals[7]  = data['dimensions']
    vals[8]  = data['hidden_units']
    vals[9]  = data['cycles']
    vals[10] = data['learning_rate']
    return vals

def split_energy_lists(results):
    '''
        Splits the list of results into compatible sublists, where each list
        only contains results with equal numbers of particles, dimensions,
        and cycles.
    '''

    conditions = {}
    ID = '{:.0f}{:.0f}{:.0f}'
    for result in results:
        key = ID.format(result[6],result[7],result[9])
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
    for i in [6,7,9]:
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
    msg2 = 'Multiple energies found for same conditions.'
    for i,(a,b) in enumerate(zip(LR, HU)):
        for j,(c,d) in enumerate(zip(a, b)):
            cond_1 = results[:,10] == c
            cond_2 = results[:,8] == d
            idx = np.where(np.logical_and(cond_1, cond_2))
            assert len(idx) == 1, msg2
            idx = idx[0]
            energy_grid_sample[i,j] = results[idx].squeeze()[0]
            energy_grid_blocking[i,j] = results[idx].squeeze()[2]
            energy_grid_bootstrap[i,j] = results[idx].squeeze()[4]

    return energy_grid_sample, energy_grid_blocking, energy_grid_bootstrap, LR, HU

def get_energy_grids(data, test = False):
    '''
        Calculates the mean energies after optimization is complete for all
        points given in a list of data dicts.
    '''
    pool = Pool()
    for i in range(len(data)):
        data[i]['test'] = test
    results = np.zeros((len(data), 11), dtype = np.float64)
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
        key = f'P{result[0][6]:.0f}D{result[0][7]:.0f}C{result[0][9]:.0f}'
        sorted_results[key] = {}
        E_sample, E_blocking, E_bootstrap, LR, HU = \
        sort_energy_grids(np.array(result))
        sorted_results[key]['sample'] = E_sample
        sorted_results[key]['blocking'] = E_blocking
        sorted_results[key]['bootstrap'] = E_bootstrap
        sorted_results[key]['LR'] = LR
        sorted_results[key]['HU'] = HU
    return sorted_results
