from pathlib import Path
import numpy as np
import os, re

project_path = Path.cwd()
data_path = project_path.parents[0] / 'Data'

def read_optimization():
    '''
        Reads files with names of formats:
            'rbm_cumulative_results_Pp_Dd_Hh_Ocycles_Eeta.csv'

        Where:
            'P' is the number of particles
            'D' is the number of dimensions
            'H' is the number of hidden units
            'O' is the number of optimization cycles
            'C' is the number of cycles
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*rbm_cumulative_results_'
    ext = r'\.csv'

    pattern_optimization_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles_\d+(?:\.\d+)?eta' + ext

    pattern_optimization_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+(?:\.\d+)?)eta' + ext

    labels_csv = ['particle', 'dimensions', 'hidden_units',
                  'cycles_optimization', 'learning_rate']
    files = {}

    for f in all_files:
        f = re.findall(pattern_optimization_1, str(f))
        if f:
            files[f[0].__str__()] = {}
            vals = re.findall(pattern_optimization_2, f[0])[0]
            for k,v in zip(labels_csv, vals):
                files[f[0].__str__()][k] = v

    for i in files.items():
        print(i)

def read_energy_samples():
    '''
        Reads files with names of formats:
            'energy_samples_Pp_Dd_Hh_Ccycles_Eeta.bin'

        Where:
            'P' is the number of particles
            'D' is the number of dimensions
            'H' is the number of hidden units
            'O' is the number of optimization cycles
            'C' is the number of cycles
            'E' is the learning rate
    '''
    all_files = list(data_path.glob(r'**/*'))
    base = r'.*energy_samples_'
    ext = r'\.bin'

    labels_csv = ['particle', 'dimensions', 'hidden_units',
                  'cycles_optimization', 'learning_rate']
    files = {}

    pattern_energies_1 = \
    base + r'\d+p_\d+d_\d+h_\d+cycles' + ext

    pattern_energies_2 = \
    base + r'(\d+)p_(\d+)d_(\d+)h_(\d+)cycles_(\d+(?:\.\d+)?)eta' + ext

    labels_bin = ['particle', 'dimensions', 'hidden_units',
                  'cycles']
    for f in all_files:
        f = re.findall(pattern_optimization_1, str(f))
        if f:
            files[f[0].__str__()] = {}
            vals = re.findall(pattern_optimization_2, f[0])[0]
            for k,v in zip(labels_csv, vals):
                files[f[0].__str__()][k] = v

    for i in files.items():
        print(i)

if __name__ == '__main__':
    read_optimization()
