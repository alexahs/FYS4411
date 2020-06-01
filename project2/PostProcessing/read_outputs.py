from pathlib import Path
import numpy as np
import os, re

project_path = Path.cwd()
data_path = project_path.parents[0] / 'Data'

def read_optimization():
    '''
        Reads files with names of formats:
            'rbm_cumulative_results_Pp_Dd_Hh_Ccycles_Eeta.csv'

        Where:
            'P' is the number of particles
            'D' is the number of dimensions
            'H' is the number of hidden units
            'C' is the number of optimization cycles
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*rbm_cumulative_results_'
    ext = r'\.csv'

    pattern_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles_\d+(?:\.\d+)?eta' + ext

    pattern_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+(?:\.\d+)?)eta' + ext

    labels = ['particle', 'dimensions', 'hidden_units', 'cycles',
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
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*energy_samples_'
    ext = r'\.bin'

    pattern_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles_\d+(?:\.\d+)?eta' + ext

    pattern_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+(?:\.\d+)?)eta' + ext

    labels = ['particle', 'dimensions', 'hidden_units', 'cycles',
              'learning_rate']

    files = []

    for f in all_files:
        match = re.findall(pattern_1, str(f))
        if match:
            files.append({'path':f})
            vals = re.findall(pattern_2, str(f))[0]
            for k,v in zip(labels, vals):
                files[-1][k] = float(v)

            files[-1]['Energy'] = np.fromfile(f, dtype = np.float64)

    return files

if __name__ == '__main__':
    optimizations = read_optimization()
    energies = read_energy_samples()

    print(optimizations)
    print(energies)