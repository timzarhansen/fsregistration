import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_command_to_batchfile(fullpath, command, append=False):
    mode = 'a' if append else 'w'
    with open(fullpath, mode) as fid:
        fid.write(f"{command}\n")


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear, clf - skipped

method_of_interest = 2
voxel_size_list = [64, 128, 256, 512]
initial_guess = 1

journal_ordner = "/home/tim-external/dataFolder/journalPaperDatasets/newDatasetsCreation/"

ordner1 = "speedTestsKeller"
ordner2 = "speedTestsValentin"

results_experiment1 = pd.read_csv(journal_ordner + ordner1 + "/results_seriell.csv", header=None).values
results_experiment2 = pd.read_csv(journal_ordner + ordner2 + "/results_seriell.csv", header=None).values

results_experiment = np.vstack([results_experiment1, results_experiment2])

for i in range(4):
    voxel_size = voxel_size_list[i]
    result_list_of_interest = results_experiment[results_experiment[:, 1] == voxel_size, :]
    result_list_of_interest_1 = result_list_of_interest[result_list_of_interest[:, 0] == 1, :]
    result_list_of_interest_1 = result_list_of_interest_1[result_list_of_interest_1[:, 2] == 1, :]

    result_list_of_interest_2 = result_list_of_interest[result_list_of_interest[:, 0] == 3, :]
    result_list_of_interest_2 = result_list_of_interest_2[result_list_of_interest_2[:, 2] == 1, :]

    result_list_of_interest_3 = result_list_of_interest[result_list_of_interest[:, 0] == 4, :]
    result_list_of_interest_3 = result_list_of_interest_3[result_list_of_interest_3[:, 2] == 1, :]

    result_list_of_interest_4 = result_list_of_interest[result_list_of_interest[:, 0] == 5, :]
    result_list_of_interest_4 = result_list_of_interest_4[result_list_of_interest_4[:, 2] == 1, :]

    result_list_of_interest_5 = result_list_of_interest[result_list_of_interest[:, 0] == 6, :]
    result_list_of_interest_5 = result_list_of_interest_5[result_list_of_interest_5[:, 2] == 1, :]

    result_list_of_interest_6 = result_list_of_interest[result_list_of_interest[:, 0] == 6, :]
    result_list_of_interest_6 = result_list_of_interest_6[result_list_of_interest_6[:, 2] == 0, :]

    result_list_of_interest_7 = result_list_of_interest[result_list_of_interest[:, 0] == 7, :]
    result_list_of_interest_7 = result_list_of_interest_7[result_list_of_interest_7[:, 2] == 1, :]

    result_list_of_interest_8 = result_list_of_interest[result_list_of_interest[:, 0] == 7, :]
    result_list_of_interest_8 = result_list_of_interest_8[result_list_of_interest_8[:, 2] == 0, :]

    result_list_of_interest_9 = result_list_of_interest[result_list_of_interest[:, 0] == 8, :]
    result_list_of_interest_9 = result_list_of_interest_9[result_list_of_interest_9[:, 2] == 1, :]

    result_list_of_interest_10 = result_list_of_interest[result_list_of_interest[:, 0] == 8, :]
    result_list_of_interest_10 = result_list_of_interest_10[result_list_of_interest_10[:, 2] == 0, :]

    result_list_of_interest_11 = result_list_of_interest[result_list_of_interest[:, 0] == 9, :]
    result_list_of_interest_11 = result_list_of_interest_11[result_list_of_interest_11[:, 2] == 0, :]

    result_list_of_interest_12 = result_list_of_interest[result_list_of_interest[:, 0] == 10, :]
    result_list_of_interest_12 = result_list_of_interest_12[result_list_of_interest_12[:, 2] == 0, :]

    result_list_of_interest_13 = result_list_of_interest[result_list_of_interest[:, 0] == 11, :]
    result_list_of_interest_13 = result_list_of_interest_13[result_list_of_interest_13[:, 2] == 0, :]

    result_list_of_interest_14 = result_list_of_interest[result_list_of_interest[:, 0] == 12, :]
    result_list_of_interest_14 = result_list_of_interest_14[result_list_of_interest_14[:, 2] == 0, :]

    result_list_of_interest_15 = result_list_of_interest[result_list_of_interest[:, 0] == 13, :]
    result_list_of_interest_15 = result_list_of_interest_15[result_list_of_interest_15[:, 2] == 0, :]

    result_list_of_interest_16 = result_list_of_interest[result_list_of_interest[:, 0] == 14, :]
    result_list_of_interest_16 = result_list_of_interest_16[result_list_of_interest_16[:, 2] == 0, :]

    result_list_of_interest_17 = result_list_of_interest[result_list_of_interest[:, 0] == 15, :]
    result_list_of_interest_17 = result_list_of_interest_17[result_list_of_interest_17[:, 2] == 0, :]

    result_list_of_interest_18 = result_list_of_interest[result_list_of_interest[:, 0] == 16, :]
    result_list_of_interest_18 = result_list_of_interest_18[result_list_of_interest_18[:, 2] == 0, :]

    plt.figure(i + 1)
    plt.rc('text', usetex=True)
    plt.boxplot([
        result_list_of_interest_1[:, 5] / 1000,
        result_list_of_interest_2[:, 5] / 1000,
        result_list_of_interest_3[:, 5] / 1000,
        result_list_of_interest_4[:, 5] / 1000,
        result_list_of_interest_5[:, 5],
        result_list_of_interest_6[:, 5],
        result_list_of_interest_7[:, 5],
        result_list_of_interest_8[:, 5],
        result_list_of_interest_9[:, 5],
        result_list_of_interest_10[:, 5],
        result_list_of_interest_11[:, 5] / 1000,
        result_list_of_interest_12[:, 5] / 1000,
        result_list_of_interest_13[:, 5] / 1000,
        result_list_of_interest_14[:, 5] / 1000,
        result_list_of_interest_15[:, 5] / 1000,
        result_list_of_interest_16[:, 5] / 1000,
        result_list_of_interest_17[:, 5] / 1000,
        result_list_of_interest_18[:, 5] / 1000
    ], labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])

    plt.yscale('log')
    plt.title(f"Voxel Size: {voxel_size}")
    plt.ylabel(r"time in ms")

    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/boxplot{voxel_size}computationSpeed'
    plt.savefig(name_of_pdf_file + '.pdf')
    add_command_to_batchfile("batchfile.sh", f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)
