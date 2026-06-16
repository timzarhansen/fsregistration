import numpy as np
import matplotlib.pyplot as plt

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

# 1 GICP Error, 2 GICP Time, 3 Super4PCS Error, 4 Super4PCS Time, 5 NDT D2D 2D Error, 6 NDT D2D 2D Time
# 7 NDT P2D Error, 8 NDT P2D Time,9 Our FMS 32 Error, 10 Our FMS 32 Time, 11 Our FMS 64 Error, 12 Our FMS 64,
# 13 Our FMS 128 Error, 14 Our FMS 128 Time, 15 Our FMS 256 Error, 16 Our FMS 256 Time
# 17 Our FMS Fast 32 Error, 18 Our FMS Fast 32 Time, 19 Our FMS Fast 64 Error, 20 Our FMS Fast 64 Time
# 21 Our FMS Fast 128 Error, 22 Our FMS Fast 128 Time, 23 Our FMS Fast 256 Error, 24 Our FMS Fast 256 Time

result_matrix = np.genfromtxt("csvFiles/comparisonAllMethodsOddAngles.csv", delimiter=",")

only_error = result_matrix[:, 0:24:2]
only_time = result_matrix[:, 1:24:2]

plt.figure(1)
plt.boxplot(only_error)
plt.gca().set_yscale('log')
plt.xticks(range(1, 13), ['GICP', 'Super4PCS', 'NDT D2D 2D', 'NDT P2D', 'Our FMS 32', 'Our FMS 64',
                           'Our FMS 128', 'Our FMS 256', 'Our FMS Fast 32', 'Our FMS Fast 64',
                           'Our FMS Fast 128', 'Our FMS Fast 256'])
plt.title("error(angle diff times norm pos diff)")

plt.figure(2)
plt.boxplot(only_time)
plt.gca().set_yscale('log')
plt.xticks(range(1, 13), ['GICP', 'Super4PCS', 'NDT D2D 2D', 'NDT P2D', 'Our FMS 32', 'Our FMS 64',
                           'Our FMS 128', 'Our FMS 256', 'Our FMS Fast 32', 'Our FMS Fast 64',
                           'Our FMS Fast 128', 'Our FMS Fast 256'])
plt.title("computation Time")
plt.ylabel("computation Time in ms")
