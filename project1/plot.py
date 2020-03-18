import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from blocking import block
plt.style.use('ggplot')

RAW_DATA_DIR = "./Data/temp_results/brute_importance_with_energies/"
FORMATTED_DATA = "./Data/formatted/"
FIGURE_DIR = "./Figures/"


if not os.path.exists(FORMATTED_DATA):
    os.makedirs(FORMATTED_DATA)

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

def plotSphericalVMC(dim, particles):
    df_ana = pd.read_csv(RAW_DATA_DIR + f"vmc_{dim}d_{particles}p_ana.csv")
    df_num = pd.read_csv(RAW_DATA_DIR + f"vmc_{dim}d_{particles}p_num.csv")

    outfile = FIGURE_DIR + f"fig_brute_vmc_task_b_{dim}d_{particles}p.pdf"

    plt.errorbar(df_num["Alpha"] + 0.005, df_num["Energy"], np.sqrt(df_num["Variance"]), label="Numerical derivative", fmt=".", capsize=3)
    plt.errorbar(df_ana["Alpha"] - 0.005, df_ana["Energy"], np.sqrt(df_ana["Variance"]), label="Analytic derivative", fmt=".", capsize=3)
    # plt.title(f"{dim} dimensions, {particles} particle(s)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.savefig(outfile)
    # plt.show()



def createTabulated(dim, particles, dt):

    # vmc_1d_100p_0dt_ana.csv
    # vmc_1d_100p_-1dt_ana.csv

    infileAna = RAW_DATA_DIR + f"vmc_{dim}d_{particles}p_{dt}dt_ana.csv"
    # infileNum = RAW_DATA_DIR + f"vmc_{dim}d_{particles}p_{dt}dt_num.csv"


    df_ana = pd.read_csv(infileAna)
    # df_num = pd.read_csv(infileNum)

    nLines = len(df_ana["Energy"])

    CPUtimeAna = df_ana[" ElapsedTimeMS"]
    # CPUtimeNum = df_num[" ElapsedTimeMS"]

    arrAna = df_ana.to_numpy()
    # arrNum = df_num.to_numpy()

    tableHeader = r"$\alpha$ & $\langle E \rangle$ & \langle E^2 \rangle & $\sigma^2$ & AcceptRatio \\" + "\n"

    labelAna = "\label{tab:task_b_ana_dim" + str(dim) + "_part" + str(particles) + "}" + "\n"
    # labelNum = "\label{tab:task_b_num_dim" + str(dim) + "_part" + str(particles) + "}" + "\n"

    captionAna = "\caption{Analytic derivative. CPU time: " + str(CPUtimeAna[0]*1e-3) + "s" + "}" + "\n"
    # captionNum = "\caption{Analytic derivative. CPU time: " + str(CPUtimeNum[0]*1e-3) + "s" + "}" + "\n"

    outfileAna = open(FORMATTED_DATA + "table_task_b_ana_dim" + str(dim) + "_part" + str(particles) + ".txt", "w")
    # outfileNum = open(FORMATTED_DATA + "table_task_b_num_dim" + str(dim) + "_part" + str(particles) + ".txt", "w")


    outfileAna.write(labelAna)
    outfileAna.write(captionAna)
    outfileAna.write(tableHeader)

    # outfileNum.write(labelNum)
    # outfileNum.write(captionNum)
    # outfileNum.write(tableHeader)

    # float_formatter = "{:.5f}".format
    # np.set_printoptions(formatter={'float_kind':float_formatter})

    for i in range(nLines):
        outfileAna.write("%.1f & %5.5f & %5.5f & %5.5f & %5.5f " %(arrAna[i,0], arrAna[i,1], arrAna[i,2], arrAna[i,3], arrAna[i,4]) + r"\\" + "\n")
        # outfileNum.write("%.1f & %5.5f & %5.5f & %5.5f & %5.5f " %(arrNum[i,0], arrNum[i,1], arrNum[i,2], arrNum[i,3], arrNum[i,4]) + r"\\" + "\n")


    outfileAna.close()
    # outfileNum.close()

def getAllVariance():

    log2Steps = 21
    particles = [1, 10, 100, 500]
    dimensions = [1, 2, 3]
    timeSteps = [-4, -3, -2, -1, 0, 1, 2]
    alphas = [2, 3, 4, 5, 6, 7, 8, 9] #1e-1

    nAlphas = len(alphas)
    nDims = len(dimensions)
    nTimeSteps = len(timeSteps)
    nParticles = len(particles)

    variances = np.zeros((nAlphas, nDims, nParticles, nTimeSteps))

    for a, alpha in enumerate(alphas):
        for d, dim in enumerate(dimensions):
            for p, particle in enumerate(particles):
                for dt, timeStep in enumerate(timeSteps):
                    filename = RAW_DATA_DIR + f"vmc_energysamples_{dim}d_{particle}p_{alpha}alpha_{timeStep}dt_2pow{log2Steps}steps.bin"
                    print("Loaded", filename[12:])
                    energySamples = np.fromfile(filename, dtype="double")
                    mean, var = block(energySamples)

                    variances[a, d, p, dt] = var

    outfile = "variance_blocking_importance_all_configs.npy"
    np.save(outfile, variances)


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


def bruteForceResults():
    dims = [1, 2, 3]
    particles = [1, 10, 100, 500]

    for d in dims:
        for p in particles:
            plotSphericalVMC(d, p)
            # createTabulated(d, p)


getAllVariance()

# bruteForceResults()

# sphericalVMC(dim, particles)
# blocking(dim, particles, log2Steps)
