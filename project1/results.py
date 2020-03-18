import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from blocking import block
plt.style.use('ggplot')

RAW_DATA_DIR = "./Data/temp_results/"
FORMATTED_DATA = "./Data/formatted/"
FIGURE_DIR = "./Figures/"


if not os.path.exists(FORMATTED_DATA):
    os.makedirs(FORMATTED_DATA)

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

def plotSphericalVMC(dim, particles):
    df_ana = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles}p_ana.csv")
    df_num = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles}p_num.csv")

    outfile = FIGURE_DIR + "task_b/" + f"fig_brute_vmc_task_b_{dim}d_{particles}p.pdf"


    plt.errorbar(df_num["Alpha"] + 0.005, df_num["Energy"], np.sqrt(df_num["Variance"]), label="Numerical derivative", fmt=".", capsize=3)
    plt.errorbar(df_ana["Alpha"] - 0.005, df_ana["Energy"], np.sqrt(df_ana["Variance"]), label="Analytic derivative", fmt=".", capsize=3)
    # plt.title(f"{dim} dimensions, {particles} particle(s)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.legend()
    plt.savefig(outfile)
    # plt.show()
    plt.clf()

def createTabulated_tasks_c_d(dim, particles, dt, variance_block):


    # vmc_1d_100p_0dt_ana.csv

    infileAna = RAW_DATA_DIR + "brute_importance_with_energies/" + f"vmc_{dim}d_{particles}p_{dt}dt_ana.csv"


    df_ana = pd.read_csv(infileAna)
    nLines = len(df_ana["Energy"])
    CPUtimeAna = df_ana["ElapsedTimeMS"]
    arrAna = df_ana.to_numpy()

    # tableHeader = r"$\alpha$ & $\langle E \rangle$ & \langle E^2 \rangle & $\sigma^2$ & AcceptRatio \\" + "\n"
    tableHeader = r"$\alpha$ & $\langle E \rangle$ & $\sigma_{mc}$ & $\sigma_{block}$ & AcceptRatio \\" + "\n"


    labelAna = "\label{tab:task_b_ana_dim" + str(dim) + "_part" + str(particles) + "_dt" +  str(dt) +"}" + "\n"

    captionAna = "\caption{Analytic derivative. CPU time: " + str(CPUtimeAna[0]*1e-3) + "s." + " Time step (log10):" + str(dt) + "}" + "\n"


    fnameAna = FORMATTED_DATA + "tasks_c_d/" + "table_task_b_ana_dim" + str(dim) + "_part" + str(particles) + "_dt" +  str(dt) +".txt"
    outfileAna = open(fnameAna, "w")

    outfileAna.write(labelAna)
    outfileAna.write(captionAna)
    outfileAna.write(tableHeader)



    stdAna = np.sqrt(arrAna[:,3])
    std_block = np.sqrt(variance_block)
    stdAna[np.isnan(stdAna)] = 0
    std_block[np.isnan(std_block)] = 0

    # "Alpha,Energy,Energy2,Variance,AcceptRatio,ElapsedTimeMS"

    # Alpha, Energy, sigma_mc, sigma_block, AcceptRatio

    for i in range(nLines):
        outfileAna.write("%.1f & %5.5f & %5.5f & %5.5f & %5.5f " \
                          %(arrAna[i,0], arrAna[i,1], stdAna[i], std_block[i], arrAna[i,4]) + r"\\" + "\n")




    outfileAna.close()

    print(fnameAna, "formatted")

def printResultsNumvsAna():

    


    return None


def createTabulated_task_b(dim, particles):

    # vmc_1d_100p_0dt_ana.csv
    # vmc_1d_100p_-1dt_ana.csv

    # infileAna = RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles}p_{dt}dt_ana.csv"
    # infileNum = RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles}p_{dt}dt_num.csv"

    infileAna = RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles}p_ana.csv"
    infileNum = RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles}p_num.csv"


    df_ana = pd.read_csv(infileAna)
    df_num = pd.read_csv(infileNum)

    nLines = len(df_ana["Energy"])

    CPUtimeAna = df_ana[" ElapsedTimeMS"]
    CPUtimeNum = df_num[" ElapsedTimeMS"]

    arrAna = df_ana.to_numpy()
    arrNum = df_num.to_numpy()

    # tableHeader = r"$\alpha$ & $\langle E \rangle$ & \langle E^2 \rangle & $\sigma^2$ & AcceptRatio \\" + "\n"
    # tableHeader = r"$\alpha$ & $\langle E \rangle$ & $\sigma_{mc}$ & $sigma_{block}$ & AcceptRatio \\" + "\n"

    tableHeader = r"$\alpha$ & $\langle E \rangle$ & $\sigma_{mc}$ & AcceptRatio \\" + "\n"

    labelAna = "\label{tab:task_b_ana_dim" + str(dim) + "_part" + str(particles) + "}" + "\n"
    labelNum = "\label{tab:task_b_num_dim" + str(dim) + "_part" + str(particles) + "}" + "\n"

    captionAna = "\caption{Analytic derivative. CPU time: " + str(CPUtimeAna[0]*1e-3) + "s" + "}" + "\n"
    captionNum = "\caption{Analytic derivative. CPU time: " + str(CPUtimeNum[0]*1e-3) + "s" + "}" + "\n"


    fnameAna = FORMATTED_DATA + "task_b/" + "table_task_b_ana_dim" + str(dim) + "_part" + str(particles) + ".txt"
    fnameNum = FORMATTED_DATA + "task_b/" + "table_task_b_num_dim" + str(dim) + "_part" + str(particles) + ".txt"
    outfileAna = open(fnameAna, "w")
    outfileNum = open(fnameNum, "w")


    outfileAna.write(labelAna)
    outfileAna.write(captionAna)
    outfileAna.write(tableHeader)

    outfileNum.write(labelNum)
    outfileNum.write(captionNum)
    outfileNum.write(tableHeader)

    # float_formatter = "{:.5f}".format
    # np.set_printoptions(formatter={'float_kind':float_formatter})

    # "Alpha,Energy,Energy2,Variance,AcceptRatio,ElapsedTimeMS"

    # print((arrAna[:,3])**(0.5))

    stdAna = np.sqrt(arrAna[:,3])
    stdNum = np.sqrt(arrNum[:,3])

    stdAna[np.isnan(stdAna)] = 0
    stdNum[np.isnan(stdNum)] = 0



    for i in range(nLines):
        outfileAna.write("%.1f & %5.5f & %5.5f & %5.5f " %(arrAna[i,0], arrAna[i,1], stdAna[i], arrAna[i,4]) + r"\\" + "\n")
        outfileNum.write("%.1f & %5.5f & %5.5f & %5.5f " %(arrNum[i,0], arrNum[i,1], stdNum[i], arrNum[i,4]) + r"\\" + "\n")


    outfileAna.close()
    outfileNum.close()

    print(fnameAna, "formatted")
    print(fnameNum, "formatted")


def blockAllEnergySamples():

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
    timeSteps = [-4, -3, -2, -1, 0, 1, 2]

    #(alpha, dim, particle, timeStep)
    # variance_block = np.load(RAW_DATA_DIR + "variance_blocking_importance_all_configs.npy")

    # for dt, timeStep in enumerate(timeSteps):
    for d, dim in enumerate(dims):
        for p, particle in enumerate(particles):
            plotSphericalVMC(dim, particle)
            # createTabulated_tasks_c_d(dim, particle, timeStep, variance_block[:, d, p, dt])





bruteForceResults()

# sphericalVMC(dim, particles)
# blocking(dim, particles, log2Steps)
