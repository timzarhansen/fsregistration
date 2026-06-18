import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

home_folder = "/home/tim-external/dataFolder/ValentinBunkerData/directoryTest/scanNumber_0/"

number_of_scan = 9

plt.figure(1)
ab = np.genfromtxt(f"{home_folder}{number_of_scan}intensity64.csv", delimiter=",")
plt.imshow(ab)

plt.figure(2)
ab = np.genfromtxt(f"{home_folder}{number_of_scan}intensityShifted64.csv", delimiter=",")
plt.imshow(ab)

plt.figure(3)
ab = np.genfromtxt(f"{home_folder}{number_of_scan}intensity128.csv", delimiter=",")
plt.imshow(ab)

plt.figure(4)
ab = np.genfromtxt(f"{home_folder}{number_of_scan}intensity256.csv", delimiter=",")
plt.imshow(ab)

plt.figure(5)
with_threshold = o3d.io.read_point_cloud(f"{home_folder}{number_of_scan}_Threashold.ply")
o3d.visualization.draw_geometries([with_threshold])

plt.figure(6)
one_value = o3d.io.read_point_cloud(f"{home_folder}{number_of_scan}_OneValue.ply")
o3d.visualization.draw_geometries([one_value])
