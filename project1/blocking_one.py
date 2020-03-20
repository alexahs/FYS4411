import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from blocking import block
import seaborn as sns
plt.style.use('ggplot')

RAW_DATA_DIR = "./Data/temp_results/"
FORMATTED_DATA = "./Data/formatted/"
FIGURE_DIR = "./Figures/"



dt = -1

file = RAW_DATA_DIR + "brute_importance_with_energies/" + "vmc_energysamples_3d_500p_5alpha_-1dt_1num_2pow20steps.bin"
x = np.load(file, allow_pickle=True)

print(block(x))
