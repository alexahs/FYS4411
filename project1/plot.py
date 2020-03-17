import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from blocking import block
plt.style.use('ggplot')

DATA_DIR = "./Data/temp_results"
dim = sys.argv[1]
particles = sys.argv[2]
log2Steps = int(sys.argv[3])

def sphericalVMC(dim, particles):
    df_ana = pd.read_csv(DATA_DIR + "brute_no_importance" + f"vmc_{dim}d_{particles}p_ana.csv")
    df_num = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_num.csv")

    plt.errorbar(df_num["Alpha"] + 0.005, df_num["Energy"], np.sqrt(df_num["Variance"]), label="numerical", fmt=".", capsize=3)
    plt.errorbar(df_ana["Alpha"] - 0.005, df_ana["Energy"], np.sqrt(df_ana["Variance"]), label="analytic", fmt=".", capsize=3)
    plt.title(f"{dim} dimensions, {particles} particle(s)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.show()

def bootstrap(dim, particles, log2Steps):
    #TODO

    return 1


def blocking(dim, particles, log2Steps):
    """
    Data should contain 2^(int x) points for blocking code to run properly
    """
    filename = DATA_DIR + f"energy_dump/vmc_energysamples_{dim}d_{particles}p_2pow{log2Steps}steps.bin"


    energySamples = np.fromfile(filename, dtype="double")


    mean, var = block(energySamples)

    std = np.sqrt(var)
    import pandas as pd
    from pandas import DataFrame
    print(" -------- Resampling results -------- \n")
    data ={'Mean':[mean], 'Var':[var], 'STDev':[std]}
    frame = pd.DataFrame(data,index=['Values'])
    print(frame)


# sphericalVMC(dim, particles)
blocking(dim, particles, log2Steps)
