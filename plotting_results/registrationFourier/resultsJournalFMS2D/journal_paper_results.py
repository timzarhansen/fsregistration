import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear, clf - skipped

method_of_interest = 6
voxel_size = 256
initial_guess = 0

journal_ordner = "/home/tim-external/dataFolder/journalPaperDatasets/"

ordner = "onlyRotationNoNoiseKeller"

results_experiment = pd.read_csv(journal_ordner + ordner + "/results.csv", header=None).values

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

print(f"percent of results not NAN: {len(result_list_of_interest_without_nans) / len(result_list_of_interest)}")
mean_error = np.mean(result_list_of_interest_without_nans[:, 3])
std_error = np.std(result_list_of_interest_without_nans[:, 3])
print(f"mean: {mean_error}")
print(f"std: {std_error}")

y = result_list_of_interest_without_nans[:, 3]
x1 = np.hstack([np.zeros_like(result_list_of_interest_without_nans[:, 6]), result_list_of_interest_without_nans[:, 6].reshape(-1, 1)])
x2 = np.hstack([np.zeros_like(result_list_of_interest_without_nans[:, 6]), initial_guess_list[:, 3].reshape(-1, 1)])

b1 = np.linalg.lstsq(x1, y, rcond=None)[0]
b2 = np.linalg.lstsq(x2, y, rcond=None)[0]

plt.figure(1)
plt.clf()
plt.hold(True)

plt.plot(result_list_of_interest_without_nans[:, 6], result_list_of_interest_without_nans[:, 3], '.')
x_range = [0, np.max(result_list_of_interest_without_nans[:, 6])]
y_range = [b1[0], b1[0] + np.max(result_list_of_interest_without_nans[:, 6]) * b1[1]]
plt.plot(x_range, y_range)
plt.title(f"L2 overlap regression: {b1[1]}")

name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/l2RegressionOverlap{initial_guess}{voxel_size}{ordner}{method_of_interest}'
plt.savefig(name_of_pdf_file + '.pdf')
subprocess.run(['pdfcrop', name_of_pdf_file + '.pdf', name_of_pdf_file + '.pdf'])

plt.figure(2)
plt.clf()
plt.hold(True)
plt.plot(initial_guess_list[:, 3], result_list_of_interest_without_nans[:, 3], '.')
x_range = [0, np.max(initial_guess_list[:, 3])]
y_range = [b2[0], b2[0] + np.max(initial_guess_list[:, 3]) * b2[1]]
plt.plot(x_range, y_range)
plt.title(f"L2 initial guess regression: {b2[1]}")

name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/l2RegressionInitialGuess{initial_guess}{voxel_size}{ordner}{method_of_interest}'
plt.savefig(name_of_pdf_file + '.pdf')
subprocess.run(['pdfcrop', name_of_pdf_file + '.pdf', name_of_pdf_file + '.pdf'])

plt.figure(3)
plt.boxplot([result_list_of_interest_without_nans[:, 3], result_list_of_interest_without_nans[:, 4]], labels=['L2 error', 'angleError'])

plt.yscale('log')
name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/boxplot{initial_guess}{voxel_size}{ordner}{method_of_interest}'
plt.savefig(name_of_pdf_file + '.pdf')
subprocess.run(['pdfcrop', name_of_pdf_file + '.pdf', name_of_pdf_file + '.pdf'])

plt.figure(4)
plt.clf()
plt.hold(True)
plt.plot(initial_guess_list[:, 4], result_list_of_interest_without_nans[:, 4], '.')
plt.title(f"angle initial guess regression: {b2[1]}")

name_of_pdf_file = f'/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdfResults/regressionAngle{initial_guess}{voxel_size}{ordner}{method_of_interest}'
plt.savefig(name_of_pdf_file + '.pdf')
subprocess.run(['pdfcrop', name_of_pdf_file + '.pdf', name_of_pdf_file + '.pdf'])
