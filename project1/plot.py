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


def plot_correlated(log2Steps):
    DATA_DIR = "./Data/correlated_bruteforce/alpha_"
    alphas = np.arange(0.2, 0.8, 0.1)

    for p in [10, 50, 100]:
        Emean = np.zeros(alphas.shape)
        Estd = np.zeros(alphas.shape)

        for i,alpha in enumerate(alphas):
            alpha_str = f"{alpha:1.3f}"
            filename = DATA_DIR + alpha_str + f"_{dim}d_{p}p_2pow{log2Steps}steps.bin"
            size = 2**20
            print(f"Blocking for alpha={alpha:1.1f}", end=", ")
            DataAnalysis = DataAnalysisClass(filename, size)
            DataAnalysis.blocking()
            Emean[i] = DataAnalysis.blockingAvg
            Estd[i] = DataAnalysis.blockingStd

        Emean /= float(particles)
        Estd /= float(particles)
        plt.errorbar(alphas, Emean, Estd, fmt=".", capsize=3, label=f"{p} particles")

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\frac{\langle E\rangle}{N}$", rotation=0)
    plt.legend()
    plt.savefig(f"./Figures/correlated_{particles}p_2pow{log2Steps}.png")
    return None


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
plot_correlated(log2Steps)
# plotInitialState()
# animate_3D(particles)


# Histogram of RNGs, just a test to verify that they're uniform and in range (0.0, 1.0)
# x = np.fromfile("./Data/random_numbers_test_3d_10p_2pow10steps.bin")
# plt.hist(x)
# plt.show()



# alpha2 = [4.79495, -0.009344, -0.0116655, -1.06681, 15.1935]
# alpha3 = [7.61478, -0.00429117, -0.0166995, -1.51494, 9.54238]
# alpha4 = [9.08412, 0.00458998, -0.112504, -4.09997, 7.99187]
# alpha5 = [12.324, 0.0141608, -0.0966263, -3.77252, 5.90905]
# alpha6 = [14.0648, -0.00654367, -0.157027, -5.71573, 5.17561]
# alpha7 = [15.3739, 0.0181476, -0.151434, -6.17339, 4.70027]
# alpha8 = [18.5588, 0.0715338, -0.295847, -8.71794, 3.91965]


# alpha800=[19.879,0.0265097,-0.241368,2.52009,3.66181]
# alpha700=[16.415,0.079911,-0.198962,1.86431,4.4347]
# alpha500=[12.9838,0.00046371,-0.23286,1.62812,5.57916]
# alpha200=[4.83961,-0.000494919,-0.037456,0.484489,15.0539]
# alpha600=[14.8137,0.0207176,-0.177458,1.54319,4.91557]
# alpha300=[7.50715,0.0160751,-0.042799,0.602857,9.69187]
# alpha400=[10.1977,0.00116509,-0.0407901,0.767628,7.12186]
#
#
#
# dat = np.array([alpha200, alpha300, alpha400, alpha500, alpha600, alpha700, alpha800])
# dat /= 10
# alpha = np.arange(0.2, 0.8, 0.1)
# plt.plot(alpha, dat[:, 0], label="term1")
# plt.plot(alpha, dat[:, 1], label="term2")
# plt.plot(alpha, dat[:, 2], label="term3")
# plt.plot(alpha, dat[:, 3], label="term4")
# plt.plot(alpha, dat[:, 4], label="potential")
# # dat[:, 3] = np.zeros(dat[:, 3].shape)
# plt.plot(alpha, np.sum(dat, axis=1), "g--",  label="Total")
# plt.legend()
# plt.show()
