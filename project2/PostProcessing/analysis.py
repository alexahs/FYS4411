"""
    UPDATED by Gabriel for python3
"""

from sys import argv
from os import mkdir, path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties
from multiprocessing import Pool
from numba import njit, prange

# Timing Decorator
def timeFunction(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s Function Took: \t %0.3f s' % (f.__name__.title(), (time2-time1)))
        return ret
    return wrap

class dataAnalysisClass:
    # General Init functions
    def __init__(self, data):
        self.avg = np.mean(data)
        self.var = np.var(data)
        self.std = np.std(data)
        self.data = data
        self.samplingErrors()

    def loadData(self, size=0):
        if size != 0:
            # MODIFIED BY GABRIEL
            pool = Pool()
            with open(self.inputFileName, 'r') as inputFile:
                self.data = inputFile.read().split()
                self.data = np.array(pool.map(float, self.data))
            """
            # ORIGINAL BY MHJ
            with open(self.inputFileName) as inputFile:
                self.data = np.zeros(size)
                for x in xrange(size):
                    self.data[x] = float(next(inputFile))
            """
        else:
            self.data = np.loadtxt(self.inputFileName)

    # Statistical Analysis with Multiple Methods
    def runAllAnalyses(self, skip = False):
        if not skip:
            self.bootAvg, self.bootVar, self.bootStd = self.bootstrap(self.data)
            self.blockingAvg, self.blockingVar, self.blockingStd = self.blocking(self.data)
        else:
            self.bootAvg, self.bootVar, self.bootStd = 0,0,0
            self.blockingAvg, self.blockingVar, self.blockingStd = 0,0,0

    # Standard Autocorrelation
    @timeFunction
    def autocorrelation(self):
        self.acf = np.zeros(len(self.data)//2)
        for k in range(0, len(self.data)//2):
            self.acf[k] = np.corrcoef(np.array([self.data[0:len(self.data)-k], \
                                            self.data[k:len(self.data)]]))[0,1]

    # Bootstrap
    @staticmethod
    @njit(cache = True, parallel = True)
    def bootstrap(data, nBoots = 1000):
        bootVec = np.zeros(nBoots)
        for k in prange(0, nBoots):
            bootVec[k] = np.mean(np.random.choice(data, len(data)))
        return np.mean(bootVec), np.var(bootVec), np.std(bootVec)

    def samplingErrors(self):
        self.std /= np.sqrt(len(self.data))

    # Jackknife
    @timeFunction
    def jackknife(self):
        jackknVec = np.zeros(len(self.data))
        for k in range(0,len(self.data)):
            jackknVec[k] = np.mean(np.delete(self.data, k))
        self.jackknAvg = self.avg - (len(self.data) - 1) * (np.mean(jackknVec) - self.avg)
        self.jackknVar = float(len(self.data) - 1) * np.var(jackknVec)
        self.jackknStd = np.sqrt(self.jackknVar)

    # Blocking
    @staticmethod
    @njit(cache = True)
    def blocking(data, blockSizeMax = 500):
        blockSizeMin = 1

        blockSizes = []
        meanVec = []
        varVec = []

        for i in range(blockSizeMin, blockSizeMax):
            if(len(data) % i != 0):
                pass#continue
            blockSize = i
            meanTempVec = []
            startPoint = 0
            endPoint = blockSize

            while endPoint <= len(data):
                meanTempVec.append(np.mean(data[startPoint:endPoint]))
                startPoint = endPoint
                endPoint += blockSize
            vectorized = np.array(meanTempVec)
            if len(meanTempVec) != 0:
                mean = np.mean(vectorized)
                var = np.var(vectorized)/len(meanTempVec)
                meanVec.append(mean)
                varVec.append(var)
            blockSizes.append(blockSize)

        arr_varVec = np.array(varVec[-200:])
        arr_meanVec = np.array(meanVec[-200:])
        blockingVar = np.mean(arr_varVec)

        return np.mean(arr_meanVec), np.mean(arr_varVec), np.sqrt(blockingVar)

    # Plot of Data, Autocorrelation Function and Histogram
    def plotAll(self):
        self.createOutputFolder()
        if len(self.data) <= 100000:
            self.plotAutocorrelation()
        self.plotData()
        self.plotHistogram()
        self.plotBlocking()

    # Create Output Plots Folder
    def createOutputFolder(self):
        self.outName = self.inputFileName[:-4]
        if not path.exists(self.outName):
            mkdir(self.outName)

    # Plot the Dataset, Mean and Std
    def plotData(self):
        self.createOutputFolder()
        # Far away plot
        font = {'fontname':'serif'}
        plt.plot(range(0, len(self.data)), self.data, 'r-', linewidth=1)
        plt.plot([0, len(self.data)], [self.avg, self.avg], 'b-', linewidth=1)
        plt.plot([0, len(self.data)], [self.avg + self.std, self.avg + self.std], 'g--', linewidth=1)
        plt.plot([0, len(self.data)], [self.avg - self.std, self.avg - self.std], 'g--', linewidth=1)
        plt.ylim(self.avg - 5*self.std, self.avg + 5*self.std)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.xlim(0, len(self.data))
        plt.ylabel(self.outName.title() + ' Monte Carlo Evolution', **font)
        plt.xlabel('MonteCarlo History', **font)
        plt.title(self.outName.title(), **font)
        plt.savefig(self.outName + "/data.eps")
        plt.savefig(self.outName + "/data.png")
        plt.clf()

    # Plot Histogram of Dataset and Gaussian around it
    def plotHistogram(self):
        self.createOutputFolder()
        binNumber = 50
        font = {'fontname':'serif'}
        count, bins, ignore = plt.hist(self.data, bins=np.linspace(self.avg - 5*self.std, self.avg + 5*self.std, binNumber))
        plt.plot([self.avg, self.avg], [0,np.max(count)+10], 'b-', linewidth=1)
        plt.ylim(0,np.max(count)+10)
        plt.ylabel(self.outName.title() + ' Histogram', **font)
        plt.xlabel(self.outName.title() , **font)
        plt.title('Counts', **font)

        #gaussian
        norm = 0
        for i in range(0,len(bins)-1):
            norm += (bins[i+1]-bins[i])*count[i]
        plt.plot(bins,  norm/(self.std * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.avg)**2 / (2 * self.std**2) ), linewidth=1, color='r')
        plt.savefig(self.outName + "/hist.eps")
        plt.savefig(self.outName + "/hist.png")
        plt.clf()

    # Plot the Autocorrelation Function
    def plotAutocorrelation(self):
        self.createOutputFolder()
        font = {'fontname':'serif'}
        plt.plot(range(1, len(self.data)//2), self.acf[1:], 'r-')
        plt.ylim(-1, 1)
        plt.xlim(0, len(self.data)//2)
        plt.ylabel('Autocorrelation Function', **font)
        plt.xlabel('Lag', **font)
        plt.title('Autocorrelation', **font)
        plt.savefig(self.outName + "/autocorrelation.eps")
        plt.savefig(self.outName + "/autocorrelation.png")
        plt.clf()

    def plotBlocking(self):
        self.createOutputFolder()
        font = {'fontname':'serif'}
        plt.plot(self.blockSizes, self.varVec, 'r-')
        plt.ylabel('Variance', **font)
        plt.xlabel('Block Size', **font)
        plt.title('Blocking', **font)
        plt.savefig(self.outName + "/blocking.eps")
        plt.savefig(self.outName + "/blocking.png")
        plt.clf()

    # Print Stuff to the Terminal
    def printOutput(self):
        print("\nSample Size:    \t", len(self.data))
        print("\n=========================================\n")
        print("Sample Average: \t", self.avg)
        print("Sample Variance:\t", self.var)
        print("Sample Std:     \t", self.std)
        print("\n=========================================\n")
        print("Bootstrap Average: \t", self.bootAvg)
        print("Bootstrap Variance:\t", self.bootVar)
        print("Bootstrap Error:   \t", self.bootStd)
        print("\n=========================================\n")
        print("Jackknife Average: \t", self.jackknAvg)
        print("Jackknife Variance:\t", self.jackknVar)
        print("Jackknife Error:   \t", self.jackknStd)
        print("\n=========================================\n")
        print("Blocking Average: \t", self.blockingAvg)
        print("Blocking Variance:\t", self.blockingVar)
        print("Blocking Error:   \t", self.blockingStd, "\n")

    def returnOutput(self):
        dict_out = {
                    'sample' :    {
                                   'avg' : self.avg,
                                   'var' : self.var,
                                   'std' : self.std
                                  },
                    'bootstrap' : {
                                   'avg' : self.bootAvg,
                                   'var' : self.bootVar,
                                   'std' : self.bootStd
                                  },
                    'blocking' : {
                                   'avg' : self.blockingAvg,
                                   'var' : self.blockingVar,
                                   'std' : self.blockingStd
                                  }
                   }
        return dict_out
