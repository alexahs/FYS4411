import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import sys
from blocking import block
from analysis import DataAnalysisClass, timeFunction


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
    alphas = np.arange(0.2, 0.8, 0.1)
    Emean = np.zeros(alphas.shape)
    Estd = np.zeros(alphas.shape)

    for i,alpha in enumerate(alphas):
        alpha_str = f"{alpha:1.3f}"
        filename = DATA_DIR + alpha_str + f"_{dim}d_{particles}p_2pow{log2Steps}steps.bin"
        size = 2**17
        print(f"Blocking for alpha={alpha:1.1f}", end=", ")
        DataAnalysis = DataAnalysisClass(filename, size)
        DataAnalysis.blocking()
        Emean[i] = DataAnalysis.blockingAvg
        Estd[i] = DataAnalysis.blockingStd

    Emean /= float(particles)
    Estd /= float(particles)
    plt.title(f"{particles} particles")
    plt.errorbar(alphas, Emean, Estd, fmt=".", capsize=3, label=r"$E\pm \sigma_E$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$ / N")
    plt.legend()
    plt.show()
    plt.show()


def plotInitialState():
    df = pd.read_csv("./Data/RandomUniform.csv")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df["x"], df["y"], df["z"], "bo")
    plt.show()


def animate_3D(particles):
    data = [] # list over particles
    frames = 20470
    for i in range(1, frames + 1):
        df = pd.read_csv(f"./Data/particles/frame{i}.csv")
        data.append(df.values)

    data = np.asarray(data)
    data = np.transpose(data, [1,2,0])
    print(data.shape)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    limit = [-1, 1]
    ax.set_xlim3d(limit)
    ax.set_ylim3d(limit)
    ax.set_zlim3d(limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Frame 0")
    plot = ax.plot(data[:,0,0], data[:,1,0], data[:,2,0], "o")

    def update_plot(num, plot, data):
        ax.clear()
        plot = ax.plot(data[:,0,num], data[:,1,num], data[:,2,num], "o")
        ax.set_xlim3d(limit)
        ax.set_ylim3d(limit)
        ax.set_zlim3d(limit)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {num}")
        return None

    ani = animation.FuncAnimation(
        fig, update_plot, frames, fargs=(plot, data), interval=8, blit=False
    )

    plt.show()
    # ani.save("/Data/particles.gif", writer="imagemagick", fps=30)


# sphericalVMC(dim, particles)
# blocking(dim, particles, log2Steps)
correlated(dim, particles, log2Steps)
# plotInitialState()
# animate_3D(particles)


# Histogram of RNGs, just a test to verify that they're uniform and in range (0.0, 1.0)
# x = np.fromfile("./Data/random_numbers_test_3d_10p_2pow10steps.bin")
# plt.hist(x)
# plt.show()
