import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

DATA_DIR = "./Data/"


def sphericalVMC():
    df_ana = pd.read_csv(DATA_DIR + "vmc_3d_10p_ana.csv")
    df_num = pd.read_csv(DATA_DIR + "vmc_3d_10p_num.csv")
    plt.errorbar(df_num["Alpha"], df_num["Energy"], np.sqrt(df_num["Variance"]), 0, label="numerical", fmt="bo")
    plt.errorbar(df_ana["Alpha"], df_ana["Energy"], np.sqrt(df_ana["Variance"]), 0, label="analytic", fmt="yo")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$<E>$")
    plt.legend()
    plt.show()


sphericalVMC()
