import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

plt.clf()

plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['legend.fontsize'] = 'medium'

factor = 1

# resultsExperiment = np.genfromtxt("/home/tim-external/timeMeasurements/valentinTest256.csv", delimiter=",")
# nameOfDataset = "Valentin256"
# whereLegend = "southeast"

resultsExperiment = np.genfromtxt("/home/tim-external/timeMeasurements/valentinTest.csv", delimiter=",")
nameOfDataset = "Valentin128"
whereLegend = "southeast"

# 1. time of optimization
# 2. oveall graph creation
# 3. type A
# 4. Type B
# 5. Scan Time

timeSinceStart = np.zeros(resultsExperiment.shape[0])
percentageOfCalcOptimization = np.zeros(resultsExperiment.shape[0] - 1)
scanTime = np.zeros(resultsExperiment.shape[0] - 1)
completeComputationTime = np.zeros(resultsExperiment.shape[0] - 1)
numberOfNodes = np.zeros(resultsExperiment.shape[0] - 1)
numberOfEdges = np.zeros(resultsExperiment.shape[0] - 1)

for i in range(1, resultsExperiment.shape[0]):
    percentageOfCalcOptimization[i - 1] = resultsExperiment[i, 0] / (resultsExperiment[i, 0] + resultsExperiment[i, 1] + resultsExperiment[i, 2] + resultsExperiment[i, 3])
    scanTime[i - 1] = resultsExperiment[i, 4]
    completeComputationTime[i - 1] = resultsExperiment[i, 0] + resultsExperiment[i, 1] + resultsExperiment[i, 2] + resultsExperiment[i, 3]
    numberOfNodes[i - 1] = resultsExperiment[i, 5]
    numberOfEdges[i - 1] = resultsExperiment[i, 6]
    timeSinceStart[i] = timeSinceStart[i - 1] + scanTime[i - 1] / 60.0

timeSinceStart = timeSinceStart[1:]


def smooth(data):
    from scipy.signal import savgol_filter
    return savgol_filter(data, min(len(data), 11), 2)


plt.figure(1)
plt.clf()
plt.hold(True)
plt.box(True)
plt.grid(True)
plt.plot(timeSinceStart, smooth(scanTime))
plt.plot(timeSinceStart, smooth(completeComputationTime))
ax = plt.axis()
plt.axis([ax[0], ax[1], -1.5, ax[3] * 1.001])

plt.ylabel("seconds")
plt.xlabel("minutes")

plt.legend(['scan acquisition time', 'overall computation time'], loc=whereLegend)

plt.gca().set_aspect(2 / (1 * 1))
nameOfPdfFile = "/home/ws/matlab/registrationFourier/ICRA2024Scripts/pdfs/computationTimes" + nameOfDataset
plt.savefig(nameOfPdfFile + '.pdf', bbox_inches='tight')

systemCommand = "pdfcrop " + nameOfPdfFile + ".pdf " + nameOfPdfFile + ".pdf"
subprocess.call(systemCommand, shell=True)

plt.figure(2)
plt.clf()
plt.hold(True)
plt.box(True)
plt.grid(True)
plt.plot(numberOfEdges, smooth(scanTime))
plt.plot(numberOfEdges, smooth(completeComputationTime))

plt.ylabel("time in s")
plt.xlabel("number Of Edges In Graph")

plt.legend(['scan acquisition time', 'overall computation time'], loc=whereLegend)

plt.gca().set_aspect(2 / (1 * 1))
nameOfPdfFile = "/home/ws/matlab/registrationFourier/ICRA2024Scripts/pdfs/computationEdges" + nameOfDataset
plt.savefig(nameOfPdfFile + '.pdf', bbox_inches='tight')

systemCommand = "pdfcrop " + nameOfPdfFile + ".pdf " + nameOfPdfFile + ".pdf"
subprocess.call(systemCommand, shell=True)

# 1. time of optimization
# 2. type A
# 3. Type B
# 4. oveall graph creation
# 5. Scan Time
howManyGraphs = 6
barStacked = np.zeros((howManyGraphs, 4))
timeForBarPlot = np.zeros(howManyGraphs)

for i in range(howManyGraphs):
    currentNumberOfInterest = round(resultsExperiment.shape[0] / howManyGraphs)
    averageTypeA = 0
    averageTypeB = 0
    averageComputationOverload = 0
    averageOptimization = 0
    averageTime = 0
    index = 0

    for j in range((i - 1) * round(resultsExperiment.shape[0] / howManyGraphs) + 1, i * round(resultsExperiment.shape[0] / howManyGraphs) + 1):
        if j > resultsExperiment.shape[0]:
            break
        if j > timeSinceStart.shape[0]:
            pass
        else:
            averageTime = timeSinceStart[j]
        averageTypeA = averageTypeA + resultsExperiment[j, 1]
        averageTypeB = averageTypeB + resultsExperiment[j, 2]
        averageComputationOverload = averageComputationOverload + resultsExperiment[j, 3]
        averageOptimization = averageOptimization + resultsExperiment[j, 0]
        index = index + 1
    timeForBarPlot[i] = averageTime
    barStacked[i, 0] = averageComputationOverload / index
    barStacked[i, 1] = averageTypeA / index
    barStacked[i, 2] = averageTypeB / index
    barStacked[i, 3] = averageOptimization / index

plt.figure(3)

plt.bar(np.arange(howManyGraphs), barStacked, 0.3, stackplot=True)

plt.legend(['rendering', 'type A', 'type B', 'iSAM2'], loc='upper left')

plt.gca().set_aspect(2 / (1.0 * 1))

ax = plt.axis()
plt.axis([ax[0], ax[1], ax[2], ax[3] * 1.1])

plt.xticks(ticks=np.arange(howManyGraphs), labels=np.round(timeForBarPlot, 0))

plt.ylabel("seconds")
plt.xlabel("minutes")

nameOfPdfFile = "/home/ws/matlab/registrationFourier/ICRA2024Scripts/pdfs/barGraph" + nameOfDataset
plt.savefig(nameOfPdfFile + '.pdf', bbox_inches='tight')

systemCommand = "pdfcrop " + nameOfPdfFile + ".pdf " + nameOfPdfFile + ".pdf"
subprocess.call(systemCommand, shell=True)
