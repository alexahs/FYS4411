import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from blocking import block
plt.style.use('ggplot')

RAW_DATA_DIR = "./Data/temp_results/brute_no_importance/"
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



def createTabulated(dim, particles):

    infileAna = RAW_DATA_DIR + f"vmc_{dim}d_{particles}p_ana.csv"
    infileNum = RAW_DATA_DIR + f"vmc_{dim}d_{particles}p_num.csv"


    df_ana = pd.read_csv(infileAna)
    df_num = pd.read_csv(infileNum)

    nLines = len(df_ana["Energy"])

    CPUtimeAna = df_ana[" ElapsedTimeMS"]
    CPUtimeNum = df_num[" ElapsedTimeMS"]

    arrAna = df_ana.to_numpy()
    arrNum = df_num.to_numpy()

    tableHeader = r"$\alpha$ & $\langle E \rangle$ & \langle E^2 \rangle & $\sigma^2$ & AcceptRatio \\" + "\n"

    labelAna = "\label{tab:task_b_ana_dim" + str(dim) + "_part" + str(particles) + "}" + "\n"
    labelNum = "\label{tab:task_b_num_dim" + str(dim) + "_part" + str(particles) + "}" + "\n"

    captionAna = "\caption{Analytic derivative. CPU time: " + str(CPUtimeAna[0]*1e-3) + "s" + "}" + "\n"
    captionNum = "\caption{Analytic derivative. CPU time: " + str(CPUtimeNum[0]*1e-3) + "s" + "}" + "\n"

    outfileAna = open(FORMATTED_DATA + "table_task_b_ana_dim" + str(dim) + "_part" + str(particles) + ".txt", "w")
    outfileNum = open(FORMATTED_DATA + "table_task_b_num_dim" + str(dim) + "_part" + str(particles) + ".txt", "w")


    outfileAna.write(labelAna)
    outfileAna.write(captionAna)
    outfileAna.write(tableHeader)

    outfileNum.write(labelNum)
    outfileNum.write(captionNum)
    outfileNum.write(tableHeader)

    # float_formatter = "{:.5f}".format
    # np.set_printoptions(formatter={'float_kind':float_formatter})

    for i in range(nLines):
        outfileAna.write("%.1f & %5.5f & %5.5f & %5.5f & %5.5f " %(arrAna[i,0], arrAna[i,1], arrAna[i,2], arrAna[i,3], arrAna[i,4]) + r"\\" + "\n")
        outfileNum.write("%.1f & %5.5f & %5.5f & %5.5f & %5.5f " %(arrNum[i,0], arrNum[i,1], arrNum[i,2], arrNum[i,3], arrNum[i,4]) + r"\\" + "\n")


    outfileAna.close()
    outfileNum.close()


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




bruteForceResults()

# sphericalVMC(dim, particles)
# blocking(dim, particles, log2Steps)
