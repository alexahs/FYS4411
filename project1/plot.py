import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from blocking import block


plt.style.use('ggplot')

DATA_DIR = "./Data/"
dim = sys.argv[1]
particles = sys.argv[2]
log2Steps = int(sys.argv[3])

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

def bootstrap(dim, particles, log2Steps):
    #TODO

    return 1



def blocking(dim, particles, log2Steps):
    """
    Data should contain 2^(int x) points for blocking code to run properly
    """
    filename = DATA_DIR + f"{dim}d_{particles}p_2pow{log2Steps}steps.bin"


    energySamples = np.fromfile(filename, dtype="double")


    mean, var = block(energySamples)

    std = np.sqrt(var)
    print(" -------- Resampling results -------- \n")
    data ={'Mean':[mean], 'Var':[var], 'STDev':[std]}
    frame = pd.DataFrame(data,index=['Values'])
    print(frame)


def correlated(dim, particles, log2Steps):
    DATA_DIR = "./Data/correlated_bruteforce/alpha_"
    alphas = np.arange(0.10, 0.75, 0.025)
    Emean = np.zeros(alphas.shape)
    Evar = np.zeros(alphas.shape)

    for i,alpha in enumerate(alphas):
        alpha_str = f"{alpha:1.3f}"
        E = np.fromfile(DATA_DIR + alpha_str + f"_{dim}d_{particles}p_2pow{log2Steps}steps.bin")
        Emean[i], Evar[i] = block(E)

    plt.title(f"{particles} particles")
    plt.errorbar(alphas, Emean, np.sqrt(Evar), fmt=".", capsize=3, label=r"$E\pm \sigma_E$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.show()
    plt.show()


def plotInitialState():
    df = pd.read_csv("./Data/RandomUniform.csv")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df["x"], df["y"], df["z"], "bo")
    plt.show()

# sphericalVMC(dim, particles)
# blocking(dim, particles, log2Steps)
# correlated(dim, particles, log2Steps)

plotInitialState()
