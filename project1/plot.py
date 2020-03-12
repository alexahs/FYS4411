import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use('ggplot')

DATA_DIR = "./Data/"
dim = sys.argv[1]
particles = sys.argv[2]
logSteps = int(sys.argv[3])

def sphericalVMC(dim, particles):
    df_ana = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_ana.csv")
    df_num = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_num.csv")

    plt.errorbar(df_num["Alpha"] + 0.005, df_num["Energy"], np.sqrt(df_num["Variance"]), label="numerical", fmt=".", capsize=3)
    plt.errorbar(df_ana["Alpha"] - 0.005, df_ana["Energy"], np.sqrt(df_ana["Variance"]), label="analytic", fmt=".", capsize=3)
    plt.title(f"{dim} dimensions, {particles} particle(s)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.show()



def resample(dim, particles, logSteps):



    filename = DATA_DIR + f"vmc_energysamples_{dim}d_{particles}p_1e{logSteps}steps.bin"

    energySamples = np.fromfile(filename, dtype="double")

    print(energySamples)



# sphericalVMC(dim, particles)
resample(dim, particles, logSteps)
