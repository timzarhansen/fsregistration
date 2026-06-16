import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_command_to_batchfile(fullpath, command, append=False):
    mode = 'a' if append else 'w'
    with open(fullpath, mode) as fid:
        fid.write(f"{command}\n")


def method_of_interest_to_string(which_method):
    method_map = {
        1: "GICP",
        2: "SUPER4PCS",
        3: "NDT D2D 2D",
        4: "NDT P2D",
        5: "FourierMellinTransform",
        6: "Our FMS 2D",
        7: "FMS hamming",
        8: "FMS none",
        9: "Feature AKAZE",
        10: "Feature KAZE",
        11: "Feature ORB",
        12: "Feature BRISK",
        13: "Feature SURF",
        14: "Feature SIFT ",
        15: "gmmRegistrationD2D",
        16: "gmmRegistrationP2D"
    }
    return method_map.get(which_method, "")


def plot_pdfs(method_of_interest, voxel_size, initial_guess, ordner, use_points, size_of_grid, number_of_combined_datasets, name_of_batch_file, ymax, plot_numbers, l2_norm_max):
    journal_ordner = "/home/tim-external/dataFolder/journalPaperDatasets/newDatasetsCreation/"

    for i in range(1, number_of_combined_datasets + 1):
        if i == 1:
            results_experiment = pd.read_csv(journal_ordner + ordner + "/results.csv", header=None).values
        else:
            results_experiment_new = pd.read_csv(journal_ordner + ordner + f"/results{i}.csv", header=None).values
            results_experiment = np.vstack([results_experiment, results_experiment_new])

    result_list_of_interest = results_experiment[results_experiment[:, 0] == method_of_interest, :]
    result_list_of_interest = result_list_of_interest[result_list_of_interest[:, 1] == voxel_size, :]
    result_list_of_interest = result_list_of_interest[result_list_of_interest[:, 2] == initial_guess, :]

    zero_overlap_results = result_list_of_interest[result_list_of_interest[:, 6] == 0, :]
    non_zero_overlap_results = result_list_of_interest[result_list_of_interest[:, 6] != 0, :]

    initial_guess_list = results_experiment[results_experiment[:, 0] == -1, :]

    initial_guess_list = initial_guess_list[result_list_of_interest[:, 3] != -1, :]
    result_list_of_interest_without_nans = result_list_of_interest[result_list_of_interest[:, 3] != -1, :]

    initial_guess_list = initial_guess_list[~np.isnan(result_list_of_interest_without_nans[:, 3]), :]
    result_list_of_interest_without_nans = result_list_of_interest_without_nans[~np.isnan(result_list_of_interest_without_nans[:, 3]), :]

    valid_percentage = len(result_list_of_interest_without_nans) / len(result_list_of_interest)
    print(f"percent of results not NAN: {valid_percentage}")

    mean_l2_error = np.mean(result_list_of_interest_without_nans[:, 3])
    std_l2_error = np.std(result_list_of_interest_without_nans[:, 3])
    print(f"mean: {mean_l2_error}")
    print(f"std: {std_l2_error}")

    mean_angle_error = np.mean(result_list_of_interest_without_nans[:, 4])
    std_angle_error = np.std(result_list_of_interest_without_nans[:, 4])
    print(f"mean: {mean_angle_error}")
    print(f"std: {std_angle_error}")

    y1 = result_list_of_interest_without_nans[:, 3]
    y2 = result_list_of_interest_without_nans[:, 4]
    x1 = np.hstack([np.zeros_like(result_list_of_interest_without_nans[:, 6]), result_list_of_interest_without_nans[:, 6].reshape(-1, 1)])
    x2 = np.hstack([np.zeros_like(result_list_of_interest_without_nans[:, 6]), initial_guess_list[:, 3].reshape(-1, 1)])
    x3 = np.hstack([np.zeros_like(result_list_of_interest_without_nans[:, 6]), initial_guess_list[:, 4].reshape(-1, 1)])

    b1 = np.linalg.lstsq(x1, y1, rcond=None)[0]
    b2 = np.linalg.lstsq(x2, y1, rcond=None)[0]
    b3 = np.linalg.lstsq(x3, y2, rcond=None)[0]

    plt.figure(1)
    plt.clf()
    plt.hold(True)

    if use_points:
        plt.plot(result_list_of_interest_without_nans[:, 6], result_list_of_interest_without_nans[:, 3], '.')
        x_range = [0, np.max(result_list_of_interest_without_nans[:, 6])]
        y_range = [b1[0], b1[0] + np.max(result_list_of_interest_without_nans[:, 6]) * b1[1]]
        plt.plot(x_range, y_range)
        plt.grid(True)
        plt.box(True)
    else:
        values, xedges, yedges = np.histogram2d(result_list_of_interest_without_nans[:, 6], result_list_of_interest_without_nans[:, 3], bins=[size_of_grid, size_of_grid])
        plt.imshow(values.T, aspect='auto', origin='lower')
        plt.colorbar()

        if len(result_list_of_interest_without_nans) > 0:
            plt.xticks(np.arange(0, size_of_grid + 1, size_of_grid * 0.2), [round(x, 2) for x in np.arange(0, np.max(result_list_of_interest_without_nans[:, 6]) + 1, np.max(result_list_of_interest_without_nans[:, 6]) * 0.2)])
            plt.yticks(np.arange(0, size_of_grid + 1, size_of_grid * 0.2), [round(x, 2) for x in np.arange(0, np.max(result_list_of_interest_without_nans[:, 3]) + 1, np.max(result_list_of_interest_without_nans[:, 3]) * 0.2)])

        plt.axis('equal')
        plt.set_cmap('log')

    plt.ylim([0, l2_norm_max])
    plt.gca().set_aspect(2 / 1, adjustable='box')

    if plot_numbers:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} L2 overlap regression: {b1[1]}")
    else:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} L2 overlap regression")

    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/l2RegressionOverlap{voxel_size}{ordner}{initial_guess}{method_of_interest}'
    plt.savefig(name_of_pdf_file + '.pdf')
    add_command_to_batchfile(name_of_batch_file, f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)

    plt.figure(2)
    plt.clf()
    plt.hold(True)

    if use_points:
        plt.plot(initial_guess_list[:, 3], result_list_of_interest_without_nans[:, 3], '.')
        x_range = [0, np.max(initial_guess_list[:, 3])]
        y_range = [b2[0], b2[0] + np.max(initial_guess_list[:, 3]) * b2[1]]
        plt.plot(x_range, y_range)
        plt.grid(True)
        plt.box(True)
    else:
        values, xedges, yedges = np.histogram2d(initial_guess_list[:, 3], result_list_of_interest_without_nans[:, 3], bins=[size_of_grid, size_of_grid])
        plt.imshow(values.T, aspect='auto', origin='lower')
        plt.colorbar()

        if len(result_list_of_interest_without_nans) > 0:
            plt.xticks(np.arange(0, size_of_grid + 1, size_of_grid * 0.2), [round(x, 2) for x in np.arange(0, np.max(initial_guess_list[:, 3]) + 1, np.max(initial_guess_list[:, 3]) * 0.2)])
            plt.yticks(np.arange(0, size_of_grid + 1, size_of_grid * 0.2), [round(x, 2) for x in np.arange(0, np.max(result_list_of_interest_without_nans[:, 3]) + 1, np.max(result_list_of_interest_without_nans[:, 3]) * 0.2)])

        plt.axis('equal')
        plt.set_cmap('log')

    plt.ylim([0, l2_norm_max])
    plt.gca().set_aspect(2 / 1, adjustable='box')

    if plot_numbers:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} L2 initial guess regression: {b2[1]}")
    else:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} L2 initial guess regression")

    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/l2RegressionInitialGuess{voxel_size}{ordner}{initial_guess}{method_of_interest}'
    plt.savefig(name_of_pdf_file + '.pdf')
    add_command_to_batchfile(name_of_batch_file, f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)

    if np.isnan(mean_l2_error):
        mean_l2_error = -1
        std_l2_error = -1
        mean_angle_error = -1
        std_angle_error = -1

    plt.figure(3)

    if plot_numbers:
        label1 = f"L2 error {mean_l2_error} +- {std_l2_error}"
        label2 = f"angleError {mean_angle_error}+-{std_angle_error}"
    else:
        label1 = "L2 error "
        label2 = "angleError "

    label3 = [label1, label2]

    plt.boxplot([result_list_of_interest_without_nans[:, 3], result_list_of_interest_without_nans[:, 4]], labels=label3)
    plt.gca().set_aspect(2 / 1, adjustable='box')

    if plot_numbers:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} valid: {valid_percentage}")
    else:
        plt.title(f"{method_of_interest_to_string(method_of_interest)}")

    plt.yscale('log')
    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/boxplot{voxel_size}{ordner}{initial_guess}{method_of_interest}'
    plt.savefig(name_of_pdf_file + '.pdf')
    add_command_to_batchfile(name_of_batch_file, f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)

    plt.figure(4)
    plt.clf()
    plt.hold(True)

    if use_points:
        plt.plot(initial_guess_list[:, 4], result_list_of_interest_without_nans[:, 4], '.')
        x_range = [0, np.max(initial_guess_list[:, 4])]
        y_range = [b3[0], b3[0] + np.max(initial_guess_list[:, 4]) * b3[1]]
        plt.plot(x_range, y_range)
        plt.grid(True)
        plt.box(True)
    else:
        values, xedges, yedges = np.histogram2d(initial_guess_list[:, 4], result_list_of_interest_without_nans[:, 4], bins=[size_of_grid, size_of_grid])
        plt.imshow(values.T, aspect='auto', origin='lower')
        plt.colorbar()

        if len(result_list_of_interest_without_nans) > 0:
            plt.xticks(np.arange(0, size_of_grid + 1, size_of_grid * 0.2), [round(x, 2) for x in np.arange(0, np.max(initial_guess_list[:, 4]) + 1, np.max(initial_guess_list[:, 4]) * 0.2)])
            plt.yticks(np.arange(0, size_of_grid + 1, size_of_grid * 0.2), [round(x, 2) for x in np.arange(0, np.max(result_list_of_interest_without_nans[:, 4]) + 1, np.max(result_list_of_interest_without_nans[:, 4]) * 0.2)])

        plt.axis('equal')
        plt.set_cmap('log')

    plt.ylim([0, ymax])
    plt.gca().set_aspect(2 / 1, adjustable='box')

    if plot_numbers:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} angle initial guess regression: {b3[1]}")
    else:
        plt.title(f"{method_of_interest_to_string(method_of_interest)} rotation error")

    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/regressionAngle{voxel_size}{ordner}{initial_guess}{method_of_interest}'
    plt.savefig(name_of_pdf_file + '.pdf')
    add_command_to_batchfile(name_of_batch_file, f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear, clf - skipped

name_of_batch_file = "batchfile.sh"
add_command_to_batchfile(name_of_batch_file, "#!/bin/bash", False)

ymax = 0.1
l2_norm_max = 8
number_of_combined_datasets = 2
method_of_interest = 6
voxel_size = 256
initial_guess = 0
use_points = 1
size_grid = 50
plot_numbers = False

list_of_ordner = [
    "highNoiseBoth"
]

lift_of_voxel_size = [256]

list_method_type = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [0, 6],
    [1, 7],
    [0, 7],
    [1, 8],
    [0, 8],
    [0, 9],
    [0, 10],
    [0, 11],
    [0, 12],
    [0, 13],
    [0, 14],
    [0, 15],
    [0, 16]
])

for i in range(len(list_method_type)):
    for j in range(len(list_of_ordner)):
        for k in range(len(lift_of_voxel_size)):
            plot_pdfs(list_method_type[i, 1], lift_of_voxel_size[k], list_method_type[i, 0],
                     list_of_ordner[j], use_points, size_grid, number_of_combined_datasets,
                     name_of_batch_file, ymax, plot_numbers, l2_norm_max)
