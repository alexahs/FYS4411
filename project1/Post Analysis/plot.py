import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Local files (expected to be in this directory)
from analysis import DataAnalysisClass, timeFunction

# Configuration / Global variables / Command line arguments
mpl.rcParams.update({"font.size": 18})
plt.style.use("ggplot")
DATA_DIR = "../Data/"
dim = sys.argv[1]
particles = sys.argv[2]
log2Steps = int(sys.argv[3])

"""
NOTE:
There are a couple of different functions here, and all of them pretty much
does its own analysis. Simply comment/uncomment which analyses to run at
the bottom. The file reading should pretty much work out of the box, given
that the data files actually exists within the ./Data/ folder.
"""


def plot_E_vs_alpha_simple_gaussian(dim, particles):
    df_ana = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_ana.csv")
    df_num = pd.read_csv(DATA_DIR + f"vmc_{dim}d_{particles}p_num.csv")

    plt.errorbar(
        df_num["Alpha"] + 0.005,
        df_num["Energy"],
        np.sqrt(df_num["Variance"]),
        label="numerical",
        fmt=".",
        capsize=3,
    )
    plt.errorbar(
        df_ana["Alpha"] - 0.005,
        df_ana["Energy"],
        np.sqrt(df_ana["Variance"]),
        label="analytic",
        fmt=".",
        capsize=3,
    )
    plt.title(f"{dim} dimensions, {particles} particle(s)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.show()


def plot_correlated(log2Steps):
    """
    Reads three data files of energy samples for 10, 50 and 100 particles.
    All files should be binary files of double precision (double / np.float64),
    and they should have filenames
        ../Data/correlated_bruteforce/alpha_XXX_Xp_2powXXsteps.bin
                                            ^  ^       ^
    E.g. alpha_200_50p_2pow20steps.bin
    where alpha=0.200, 50 particles and 2**20 steps (samples).

    Params:
        log2steps : int, e.g. 20 for 2**20 = 1048576.
    Returns:
        None : but saves plot as a .png file AND saves a backup of the array
               with all Emean and Estd, obtained by blocking.
    """
    DIR = DATA_DIR + "correlated_bruteforce/"
    alphas = np.arange(0.2, 0.8, 0.1)
    ALLDATA = np.zeros(
        (7, 2, 3)
    )  # Shape of array (7 alphas, 2=[Emean, Estd], 3 particle-numbers)

    # Loop over particle numbers
    for j, p in enumerate([10, 50, 100]):
        Emean = np.zeros(alphas.shape)
        Estd = np.zeros(alphas.shape)

        # Perform blocking on all alphas
        for i, alpha in enumerate(alphas):
            alpha_str = f"alpha_{alpha:1.3f}"
            filename = alpha_str + f"_{dim}d_{p}p_2pow{log2Steps}steps.bin"
            size = 2 ** 20
            print(f"Blocking for alpha={alpha:1.1f}", end=", ")
            DataAnalysis = DataAnalysisClass(DIR + filename, size)
            # Perform the blocking, NOTE: this takes around 30 seconds EACH time,
            # if we use 2**20 samples, so 10-11 minutes in total.
            DataAnalysis.blocking()
            Emean[i] = DataAnalysis.blockingAvg
            Estd[i] = DataAnalysis.blockingStd

        # We wish to have these quantities "per particle"
        Emean /= float(p)
        Estd /= float(p)
        ALLDATA[:, 0, j] = Emean
        ALLDATA[:, 1, j] = Estd
        plt.errorbar(alphas, Emean, Estd, fmt="o--", capsize=3, label=f"{p} particles")

    # Plot configuration
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\frac{\langle E\rangle}{N}$", rotation=0)
    plt.legend()
    np.savez(
        "./Data/backup/blocking_correlated_bruteforce.npz", ALLDATA
    )  # Save a backup of the array
    plt.savefig(f"./Figures/correlated_with_blocking.png")  # Save the figure
    return None


def animate_3D():
    """
    A simple animation function which just animates the particles and how they
    evolve. NB: not recommended to use a very high number of frames.
    The purpose of this function was just to verify that the particles
        (1) actually moved around
        (2) did not do anything strange (e.g. numerical instabilities)

    The datafiles are expected to be .csv where one time frame (one file) has
    the position of all particles like this
                    x,y,z
    (particle 0)    0.2,-0.3,1.1
    (particle 1)    -0.4,0.1,-0.1
    :
    (particle n-1)

    """
    data = []  # list over particles
    frames = 20000
    for i in range(1, frames + 1, 50):
        df = pd.read_csv(f"../Data/particles/frame{i}.csv")
        data.append(df.values)
    data = np.asarray(data)
    data = np.transpose(data, [1, 2, 0])
    fig = plt.figure()
    ax = Axes3D(fig)
    limit = [-2.5, 2.5]
    ax.set_xlim3d(limit)
    ax.set_ylim3d(limit)
    ax.set_zlim3d(limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Frame 0")
    plot = ax.plot(data[:, 0, 0], data[:, 1, 0], data[:, 2, 0], "o")

    def update_plot(num, plot, data):
        ax.clear()
        plot = ax.plot(data[:, 0, num], data[:, 1, num], data[:, 2, num], "o")
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
    # plt.show()
    ani.save("../Figures/particles.gif", writer="imagemagick", fps=30)


def plot_one_body_density():
    """
    Plot the radial one-body density based on discrete bins with recordings
    from long simulations of 2**20 samples. The positions were recorded to
    these bins. The files are loaded from the directory ./Data/onebodydensity

    Returns: None, but plots and shows the result.
    """
    max = 4.0
    DIR = "../Data/onebodydensity/"
    fns = ["Jastrow.csv", "NoJastrow.csv"]
    labs = ["Correlation", "Without correlation"]
    fmts = ["--", ":"]
    for fn, label, fmt in zip(fns, labs, fmts):
        df = pd.read_csv(DIR + fn)
        px, py, pz = df["x"].values, df["y"].values, df["z"].values
        bins = px.shape[0]
        num = int(bins / 2)
        pr = np.zeros(num)
        r = np.linspace(0, max, num)
        total = 0
        for i in range(num):
            # We have to do this "strange" couting method because we defined
            # the bins to start from -4, hence the middle indices will actually
            # be where the radius is the least.
            bin = (px[i] + px[-1 - i])**2  # Add from both sides of the origin
            bin += (py[i] + py[-1 - i])**2
            bin += (pz[i] + pz[-1 - i])**2
            rval = np.sqrt(bin)
            total += rval  # Scale factor so that integral sums up to 1
            pr[-i - 1] = rval
        pr /= total
        plt.plot(r, pr, fmt, label=label)
    plt.legend()
    plt.xlabel(r"$|r|$")
    plt.ylabel(r"$p(r)$", rotation=0)
    plt.show()
    return None


if __name__ == "__main__":
    # plot_E_vs_alpha_simple_gaussian(dim, particles)
    # plot_correlated(log2Steps)
    # animate_3D()
    # plot_one_body_density()
