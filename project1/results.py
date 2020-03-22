import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from blocking import block
import seaborn as sns
plt.style.use('ggplot')

RAW_DATA_DIR = "./Data/temp_results/"
FORMATTED_DATA = "./Data/formatted/"
FIGURE_DIR = "./Figures/"


if not os.path.exists(FORMATTED_DATA):
    os.makedirs(FORMATTED_DATA)

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)




def plotGD():
    dataDir = "./Data/correlated_gd/"

    n_alpha0 = 8

    # alpha_0_arr = np.array([i*0.1 for i in range(2, 10)])
    alpha_0_arr = []
    print(alpha_0_arr)


    arr_alpha_opt = np.zeros(n_alpha0)

    for i, alpha in enumerate(alpha_0_arr):

        gd_alphas = np.fromfile(dataDir + f"alpha_{alpha:0.3f}_10p_2pow17steps.bin", dtype="double")
        alpha_0_arr.append[gd_alphas[0]]
        iters = range(0, len(gd_alphas))
        plt.plot(iters, gd_alphas, label=r"$\alpha_0=$" + f"{alpha:0.1f}")
        arr_alpha_opt[i] = gd_alphas[-1]

    print("last alphas:")
    print(arr_alpha_opt)
    alpha_opt = np.mean(arr_alpha_opt)
    print("mean alpha: ", alpha_opt)

    plt.plot([0, 100], [alpha_opt, alpha_opt] , "r--", label=r"$\overline{\alpha_{opt}}=$" + f"{alpha_opt:0.5f}")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha$")
    plt.show()

# plotGD()

def lr():

    decay = 0.0005
    learningRate = 0.001
    etas = []
    for i in range(50):
        etas.append(learningRate)
        learningRate /= 1 + decay*i

    plt.semilogy(range(len(etas)), etas)
    plt.show()

lr()




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


def printBlockStd():

    # std_block_1_10_100_num = np.sqrt(np.load(RAW_DATA_DIR + "variances_blocking_1_10_100p_3d_1num.npy"))
    # std_block_1_10_100_ana = np.sqrt(np.load(RAW_DATA_DIR + "variances_blocking_1_10_100p_3d_0num.npy"))
    # std_block_500_num = np.sqrt(np.load(RAW_DATA_DIR + "variances_blocking_500p_3d_1num.npy"))
    # std_block_500_ana = np.sqrt(np.load(RAW_DATA_DIR + "variances_blocking_500p_3d_0num.npy"))
    #
    # print(std_block_1_10_100_num)
    # print(std_block_500_num)

    dt = -1

    file = RAW_DATA_DIR + "brute_importance_with_energies/" + "vmc_energysamples_3d_500p_5alpha_-1dt_1num_2pow20steps.bin"
    x = np.fromfile(file, dtype="double")

    print(block(x))



def subPlotsShperical():


    fig = plt.figure()

    fig, axes = plt.subplots(4, 2)

    # axis = (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7)

    particles = [1, 10, 100, 500]
    dims = [2, 3]

    print(axes.shape)

    for i, p in enumerate(particles):
        for j, d in enumerate(dims):

            df_ana = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{d}d_{p}p_ana.csv")
            df_num = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{d}d_{p}p_num.csv")


            axes[i, j].errorbar(df_num["Alpha"] + 0.005, df_num["Energy"], np.sqrt(df_num["Variance"]), fmt=".", capsize=3)
            axes[i, j].errorbar(df_ana["Alpha"] - 0.005, df_ana["Energy"], np.sqrt(df_ana["Variance"]), fmt=".", capsize=3)

            axes[i, j].legend([f"N={p}, d={d}"], loc='upper center')

            # if i != 3:
            #     axes[i, j].set_xticks([])



    # df_ana1 = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles[1]}p_ana.csv")
    # df_num1 = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles[1]}p_num.csv")
    # df_ana2 = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles[2]}p_ana.csv")
    # df_num2 = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles[2]}p_num.csv")
    # df_ana3 = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles[3]}p_ana.csv")
    # df_num3 = pd.read_csv(RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{particles[3]}p_num.csv")

    # outfile = FIGURE_DIR + "task_b/" + f"fig_brute_vmc_task_b_{dim}d_all_p.pdf"



    # ax1.errorbar(df_num1["Alpha"] + 0.005, df_num1["Energy"], np.sqrt(df_num1["Variance"]), label="Numerical derivative", fmt=".", capsize=3)
    # ax1.errorbar(df_ana1["Alpha"] - 0.005, df_ana1["Energy"], np.sqrt(df_ana1["Variance"]), label="Analytic derivative", fmt=".", capsize=3)
    # ax2.errorbar(df_num2["Alpha"] + 0.005, df_num2["Energy"], np.sqrt(df_num2["Variance"]), label="Numerical derivative", fmt=".", capsize=3)
    # ax2.errorbar(df_ana2["Alpha"] - 0.005, df_ana2["Energy"], np.sqrt(df_ana2["Variance"]), label="Analytic derivative", fmt=".", capsize=3)
    # ax3.errorbar(df_num3["Alpha"] + 0.005, df_num3["Energy"], np.sqrt(df_num3["Variance"]), label="Numerical derivative", fmt=".", capsize=3)
    # ax3.errorbar(df_ana3["Alpha"] - 0.005, df_ana3["Energy"], np.sqrt(df_ana3["Variance"]), label="Analytic derivative", fmt=".", capsize=3)


    # axes[0, 0].legend()
    axes[3, 0].set_xlabel(r"$\alpha$")
    axes[3, 1].set_xlabel(r"$\alpha$")
    # ax0.set_ylabel(r"$\langle E\rangle$")
    axes[0, 0].set_ylabel(r"$\langle E\rangle$")
    axes[0, 1].set_ylabel(r"$\langle E\rangle$")
    # ax2.set_ylabel(r"$\langle E\rangle$")


    plt.show()

def timeDepTable():

    inDir = "brute_importance_with_energies/"

    # vmc_3d_10p_-2dt_importance_0ana

    particle = 100
    dim = 3


    particles = np.array([1, 10, 100, 500])
    dims = [1, 2, 3]
    timesteps = [-4, -3, -2, -1, 0, 1, 2]
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    d = dim -1
    p = np.where(particles == particle)[0]
    print(p)

    var_block = np.load(RAW_DATA_DIR + "variance_blocking_importance_all_configs.npy")
    #(alpha, dim, particles, timesteps))

    outfilename = FORMATTED_DATA + "tasks_c_d/" + f"dt_dependence_{particle}p_{dim}d.txt"
    outfile = open(outfilename, "w")

    # tableHeader = r"$\alpha$ & $\langle E \rangle$ & $\sigma_{mc}$ & $\sigma_{block}$ & AcceptRatio \\" + "\n"
    tableHeader = r"$\Delta t$ & \alpha_{min} & $\langle E_{min} \rangle$  & $\sigma_{mc}$ & $\sigma_{block}$ & ratio \\" + "\n"
    outfile.write(tableHeader)

    for t, dt in enumerate(timesteps):
        df = pd.read_csv(RAW_DATA_DIR + inDir + f"vmc_{dim}d_{particle}p_{dt}dt_ana.csv")
        E_min = df["Energy"].min()
        alpha = df["Alpha"][df.Energy == E_min].to_numpy()[0]
        alphaIdx = np.where(alphas == alpha)[0][0]
        ratio = df["AcceptRatio"][df.Energy == E_min].to_numpy()[0]
        std_mc = np.sqrt(df["Variance"][df.Energy == E_min].to_numpy()[0])
        std_block = np.sqrt(var_block[alphaIdx, d, p, t])[0]

        outfile.write(f" {dt} & {alpha} & {E_min} & {std_mc} & {std_block} & {ratio}" + r" \\" + "\n")


        print(alphaIdx)

        print(" ")
        print(f"-----dt = {dt}-----")
        print(f" * E_min = {E_min}")
        print(f" * Alpha = {alpha}")
        print(f" * ratio = {ratio}")
        print(f" * std_mc = {std_mc}")
        print(f" * std_block = {std_block}")
        print(" ")
        # print(E_min)
        # print(alpha_min)

    outfile.close()
    print("File written to ", outfilename)


# timeDepTable()

def timeDep():
    inDir = "brute_importance_with_energies/"
    blocking_data = np.load(RAW_DATA_DIR + "variance_blocking_importance_all_configs.npy")
    #(alpha, dim, particles, timesteps))
    timeSteps = [-4, -3, -2, -1, 0, 1]
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphaIdx = [0, 1, 2, 3, 4, 5, 6, 7]
    dims = [1, 2, 3]
    particles = [1, 10, 100]

    block_vars = np.zeros(len(timeSteps))
    mc_vars = np.zeros(len(timeSteps))
    ratios = np.zeros(len(timeSteps))

    # CPU =


    fig, ax1 = plt.subplots()

    d = 2
    # n = 1
    aIdx = 3
    dim = dims[d]
    # particle = particles[n]
    alpha = alphas[aIdx]


    ratios = np.zeros(len(timeSteps))
    ratios_500 = np.zeros((len(timeSteps)))
    for i, dt in enumerate(timeSteps):

        for p, particle in enumerate(particles):
            df = pd.read_csv(RAW_DATA_DIR + inDir + f"vmc_{dim}d_{particle}p_{dt}dt_ana.csv")
            ratios[i] += df[df.Alpha == alpha]["AcceptRatio"]

        df_500 = pd.read_csv(RAW_DATA_DIR + inDir + f"vmc_{dim}d_500p_{dt}dt_ana.csv")
        ratios_500[i] = df_500[df_500.Alpha == alpha]["AcceptRatio"]
            # arr_ratios[p, i] = df[df.Alpha == alpha]["AcceptRatio"]

    ratios /= len(particles)
    plt.plot(timeSteps, ratios_500, label="n = 500")
    # plt.scatter(timeSteps, ratios_500)
    plt.plot(timeSteps, ratios, label="n = 1, 10, 100")
    # plt.scatter(timeSteps, ratios)
    plt.xlabel(r"Time step $\log_{10}\Delta t$")
    plt.ylabel("Ratio of accepted moves")
    plt.legend(loc=1)
    print(ratios)
    plt.show()



def timingStandardVsImport():
    inDirImportance = "brute_importance_with_energies/"
    inDirStandard = "brute_no_importance/"

    particle = 100
    dim = 3
    dt = -1

    particles = np.array([1, 10, 100, 500])
    dims = [1, 2, 3]
    timesteps = np.array([-4, -3, -2, -1, 0, 1, 2])
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    d = dim -1
    p = np.where(particles == particle)[0][0]
    t = np.where(timesteps == dt)[0][0]



    var_block = np.load(RAW_DATA_DIR + "variance_blocking_importance_all_configs.npy")
    #(alpha, dim, particles, timesteps))

    outfilename = FORMATTED_DATA + "tasks_c_d/" + f"block_and_import_vs_standard_{particle}p_{dim}d_{dt}dt.txt"
    outfile = open(outfilename, "w")

    # tableHeader = r"$\alpha$ & $\langle E \rangle$ & $\sigma_{mc}$ & $\sigma_{block}$ & AcceptRatio \\" + "\n"
    # tableHeader = r"$\Delta t$ & \alpha_{min} & $\langle E_{min} \rangle$  & $\sigma_{mc}$ & $\sigma_{block}$ & ratio \\" + "\n"
    tableHeader = r" $\alpha$ & $\langle E \rangle$ & $\sigma_{mc}$ & $\sigma_{block} & ratio_{importance} $ & ratio$_{standard}$ \\" + "\n"

    outfile.write(tableHeader)

    df1 = pd.read_csv(RAW_DATA_DIR + inDirImportance + f"vmc_{dim}d_{particle}p_{dt}dt_ana.csv")
    df2 = pd.read_csv(RAW_DATA_DIR + inDirStandard + f"vmc_{dim}d_{particle}p_ana.csv")

    for i, alpha in enumerate(alphas):
        alpha = df1["Alpha"][df1.Alpha == alpha].to_numpy()[0]
        alphaIdx = np.where(alphas == alpha)[0][0]
        ratio_imp = df1["AcceptRatio"][df1.Alpha == alpha].to_numpy()[0]
        std_mc = np.sqrt(df1["Variance"][df1.Alpha == alpha].to_numpy()[0])
        std_block = np.sqrt(var_block[alphaIdx, d, p, t])
        energy = df1["Energy"][df1.Alpha == alpha].to_numpy()[0]

        ratio_standard = df2["AcceptRatio"][df2.Alpha == alpha].to_numpy()[0]

        # float_formatter = "{:.4f}".format
        # np.set_printoptions(formatter={'float_kind':float_formatter})
        outfile.write("%.2f & %.4f & %.4f& %.4f& %.4f& %.4f" \
                        %(alpha, energy, std_mc, std_block, ratio_imp, ratio_standard) + r"\\" + "\n")

        # outfile.write(f"{alpha} & {energy} & {std_mc} & {std_block} & {ratio_imp} & {ratio_standard}" + r"\\" + "\n")



    # outfile.write(f" {dt} & {alpha} & {E_min} & {std_mc} & {std_block} & {ratio}" + r" \\" + "\n")


    outfile.close()
    print("File written to ", outfilename)


# "Alpha,Energy,Energy2,Variance,AcceptRatio,ElapsedTimeMS"
# timeDep()

def plotTimeStepDependence():

    N = 500
    dim = 1

    inDir = "brute_importance_with_energies/"

    timeSteps = [-4, -3, -2, -1, 0, 1, 2]
    n = len(timeSteps)

    var_block = np.load(RAW_DATA_DIR + "variance_blocking_importance_all_configs.npy")
    #(alpha, dim, particles, timesteps))
    fig, ax2 = plt.subplots()

    for alphaIdx, alpha in zip([0, 1, 2, 3, 4, 5, 6, 7, 8], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        energies = np.zeros(n)
        std_block = np.zeros(n)
        std_mc = np.zeros(n)
        ratio = np.zeros(n)

        #energy, std_block, accept
        for t, timeStep in enumerate(timeSteps):
            df_ana = pd.read_csv(RAW_DATA_DIR + inDir + f"vmc_{dim}d_{N}p_{timeStep}dt_ana.csv")
            df_ana = df_ana[df_ana.Alpha == alpha]

            energies[t] = df_ana["Energy"]
            ratio[t] = df_ana["AcceptRatio"]
            std_block = var_block[alphaIdx, dim-1, 2, t]


        # fig, ax1 = plt.subplots()

        # color = 'tab:red'
        # ax1.set_xlabel('time step')
        # ax1.set_ylabel('Energy', color=color)
        # ax1.errorbar(timeSteps, energies, std_block, color=color, fmt=".", capsize=3)
        # ax1.tick_params(axis='y', labelcolor=color)


        # ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('ratio', color=color)  # we already handled the x-label with ax1
        ax2.plot(timeSteps, ratio, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# plotTimeStepDependence()

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

    dim = 3

    fname = FORMATTED_DATA + "task_b/" + "ana_vs_num_all_particles_alphaHalf.txt"
    outfile = open(fname, "w")

    for p in (1, 10, 100, 500):

        infileAna = RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{p}p_ana.csv"
        infileNum = RAW_DATA_DIR + "brute_no_importance/" + f"vmc_{dim}d_{p}p_num.csv"

        df_ana = pd.read_csv(infileAna)
        df_num = pd.read_csv(infileNum)

        df_ana = df_ana[df_ana.Alpha == 0.5]
        df_num = df_num[df_num.Alpha == 0.5]

        arrAna = df_ana.to_numpy().reshape(6)
        arrNum = df_num.to_numpy().reshape(6)

        CPUtimeAna = (df_ana[" ElapsedTimeMS"])*1e-3
        CPUtimeNum = (df_num[" ElapsedTimeMS"])*1e-3



        stdAna = np.sqrt(arrAna[3])
        stdNum = np.sqrt(arrNum[3])

        #E, std, ratio, cpu || E, std, ratio, cpu
        # "Alpha,Energy,Energy2,Variance,AcceptRatio,ElapsedTimeMS"

        outfile.write("%5.5f & %5.5f & %5.5f & %5.5f  & &  %5.5f & %5.5f & %5.5f & %5.5f " \
            %(arrAna[1], stdAna, arrAna[4], CPUtimeAna, arrNum[1], stdNum, arrNum[4], CPUtimeNum) + r"\\" + "\n")
        #
    outfile.close()
    print("file written to ", fname)



def blockAllEnergySamples():
    inDir = "brute_importance_with_energies/"
    log2Steps = 21
    particles = [500]
    dimensions = [3]
    timeSteps = [-4]
    alphas = [5] #1e-1

    num = 0

    nAlphas = len(alphas)
    nDims = len(dimensions)
    nTimeSteps = len(timeSteps)
    nParticles = len(particles)

    variances = np.zeros((nAlphas, nDims, nParticles, nTimeSteps))
    # vmc_energysamples_3d_1p_9alpha_-1dt_1num_2pow21steps.bin
    # for num in [0, 1]:
    for a, alpha in enumerate(alphas):
        for d, dim in enumerate(dimensions):
            for p, particle in enumerate(particles):
                for dt, timeStep in enumerate(timeSteps):
                    filename = RAW_DATA_DIR + inDir + f"vmc_energysamples_{dim}d_{particle}p_{alpha}alpha_{timeStep}dt_{num}num_2pow{log2Steps}steps.bin"
                    print("Loaded", filename[12:])
                    energySamples = np.fromfile(filename, dtype="double")
                    mean, var = block(energySamples)

                    variances[a, d, p, dt] = var


    outfile = RAW_DATA_DIR + f"variances_blocking_500p_3d_{num}num.npy"
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
    variance_block = np.load(RAW_DATA_DIR + "variance_blocking_importance_all_configs.npy")

    # for dt, timeStep in enumerate(timeSteps):
    for d, dim in enumerate(dims):
        for p, particle in enumerate(particles):
            # plotSphericalVMC(dim, particle)
            createTabulated_tasks_c_d(dim, particle, timeStep, variance_block[:, d, p, dt])

# subPlotsShperical()
# printResultsNumvsAna()
# plotTimeStepDependence()

# bruteForceResults()

# sphericalVMC(dim, particles)
# blocking(dim, particles, log2Steps)
