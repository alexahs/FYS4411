import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use('ggplot')

DATA_DIR = "./Data/"
dim = sys.argv[1]
particles = sys.argv[2]

def sphericalVMC(dim, particles):
    df_ana = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_ana.csv")
    df_num = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_num.csv")
    plt.errorbar(df_num["Alpha"], df_num["Energy"], np.sqrt(df_num["Variance"]), 0, label="numerical", fmt="bo")
    plt.errorbar(df_ana["Alpha"], df_ana["Energy"], np.sqrt(df_ana["Variance"]), 0, label="analytic", fmt="yo")
    plt.title(f"{dim} dimensions, {particles} particle(s)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.show()


sphericalVMC(dim, particles)
