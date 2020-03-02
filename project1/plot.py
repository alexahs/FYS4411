import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./Data/"


def sphericalVMC():
    df = pd.read_csv(DATA_DIR + "vmc_3d_1p.csv")
    alpha = df["Alpha"]
    energy_num = df["Energy"]
    ri_2 = df["SumRiSquared"]
    dim = 3
    particles = 1
    omega = 1
    energy_ana = particles*dim*alpha - 2*alpha**2*ri_2 + 0.5*omega**2*ri_2
    plt.errorbar(alpha, energy_num, np.sqrt(df["Variance"]), 0, label="numerical", fmt="bo")
    plt.plot(alpha, energy_ana, "y--", label="analytic")
    plt.legend()
    plt.show()


sphericalVMC()
