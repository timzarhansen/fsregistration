import numpy as np
import pandas as pd

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

consecutive_keller = pd.read_csv("consecutiveScansKeller3_0.csv", header=None).values
consecutive_top = pd.read_csv("consecutiveScansTop3_0.csv", header=None).values

methods = np.array([1, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16])

resulting_percentages_5 = read_stuff(methods, consecutive_keller, consecutive_top, 5)


def read_stuff(methods, consecutive_keller, consecutive_top, accuracy):
    resulting_percentages_1 = np.zeros((len(methods), 2))
    consecutive_keller = consecutive_keller[consecutive_keller[:, 2] == accuracy, :]
    for i in range(consecutive_keller.shape[0]):
        index_of_method = np.where(methods == consecutive_keller[i, 0])[0]
        if len(index_of_method) > 0:
            resulting_percentages_1[index_of_method[0], 0] += consecutive_keller[i, 1]
            resulting_percentages_1[index_of_method[0], 1] += 1

    for i in range(consecutive_top.shape[0]):
        index_of_method = np.where(methods == consecutive_top[i, 0])[0]
        if len(index_of_method) > 0:
            resulting_percentages_1[index_of_method[0], 0] += consecutive_top[i, 1]
            resulting_percentages_1[index_of_method[0], 1] += 1

    for i in range(resulting_percentages_1.shape[0]):
        method = methods[i]
        percentage = resulting_percentages_1[i, 0] / resulting_percentages_1[i, 1]

    return resulting_percentages_1
