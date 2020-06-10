from pathlib import Path
import numpy as np
import os, re

project_path = Path.cwd()
data_path = project_path.parents[0] / 'Data'

def read_optimization():
    '''
        Reads files with names of formats:
            'rbm_cumulative_results_Pp_Dd_Hh_Ccycles_Ss_Eeta.csv'

        Where:
            'P' is the number of particles
            'D' is the number of dimensions
            'H' is the number of hidden units
            'C' is the number of optimization cycles
            'S' is the selected sampling method
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*rbm_cumulative_results_'
    ext = r'\.csv'

    pattern_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles_\d+s_\d+(?:\.\d+)?eta' + ext

    pattern_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+)s_(\d+(?:\.\d+)?)eta' + ext

    labels = ['particle', 'dimensions', 'hidden_units', 'cycles', 'sampling',
              'learning_rate']
    files = []

    for f in all_files:
        match = re.findall(pattern_1, str(f))
        if match:
            files.append({'path':f})
            vals = re.findall(pattern_2, str(f))[0]
            for k,v in zip(labels, vals):
                files[-1][k] = float(v)

            data = []
            with open(f, 'r') as infile:
                for n,line in enumerate(infile.readlines()):
                    if n == 0:
                        columns = line.strip().split(',')
                    else:
                        data.append(line.strip().split(','))

            data = np.array(data, dtype = np.float64).T
            for k,v in zip(columns, data):
                files[-1][k] = v
    return files

def read_energy_samples():
    '''
        Reads files with names of formats:
            'energy_samples_Pp_Dd_Hh_Ccycles_Eeta.bin'

        Where:
            'P' is the number of particles
            'D' is the number of dimensions
            'H' is the number of hidden units
            'C' is the number of cycles
            'S' is the selected sampling method
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*energy_samples_'
    ext = r'\.bin'

    pattern_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles_\d+s_\d+(?:\.\d+)?eta' + ext

    pattern_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+)s_(\d+(?:\.\d+)?)eta' + ext

    labels = ['particle', 'dimensions', 'hidden_units', 'cycles', 'sampling',
              'learning_rate']

    files = []
    defaults = -1E-3*np.ones(10)
    empty_arr = np.array([])

    for f in all_files:
        match = re.findall(pattern_1, str(f))
        if match:
            files.append({'path':f})
            vals = re.findall(pattern_2, str(f))[0]
            for k,v in zip(labels, vals):
                files[-1][k] = float(v)

            data = np.fromfile(f, dtype = np.float64)
            data = data[np.isfinite(data)]
            data = data[data != 0]
            if np.array_equal(empty_arr, data):
                data = defaults.copy()
            files[-1]['Energy'] = data

    return files

def read_pos_samples():
    '''
        Reads files with names of formats:
            'pos_samples_Pp_Dd_Hh_Ccycles_Eeta.bin'

        Where:
            'P' is the number of particles
            'D' is the number of dimensions
            'H' is the number of hidden units
            'C' is the number of cycles
            'S' is the selected sampling method
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*pos_samples_'
    ext = r'\.bin'

    pattern_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles_\d+s_\d+(?:\.\d+)?eta' + ext

    pattern_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+)s_(\d+(?:\.\d+)?)eta' + ext

    labels = ['particle', 'dimensions', 'hidden_units', 'cycles', 'sampling',
              'learning_rate']

    files = []
    defaults = -1E-3*np.ones(10)
    empty_arr = np.array([])
    for f in all_files:
        match = re.findall(pattern_1, str(f))
        if match:
            files.append({'path':f})
            vals = re.findall(pattern_2, str(f))[0]
            for k,v in zip(labels, vals):
                files[-1][k] = float(v)

            data = np.fromfile(f, dtype = np.float64)
            shape = data.shape[0]
            new_shape = (shape/files[-1]['dimensions'], files[-1]['dimensions'])
            new_shape = (int(new_shape[0]), int(new_shape[1]))
            files[-1]['pos'] = data.reshape(new_shape)
    print(files)
    return files

if __name__ == '__main__':
    optimizations = read_optimization()
    energies = read_energy_samples()
    pos = read_pos_samples()
