import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

map_data = pd.read_csv("completeMapTest.csv", header=None).values

N = int(np.sqrt(map_data.shape[0]))
map_data = np.flip(map_data, axis=1)

fig = plt.figure(4, figsize=(8, 8))
plt.imshow(map_data)
plt.ylabel("120 m")
plt.xlabel("120 m")
plt.box(on=True)

name_of_map = "ValentinourGlobal256map"
name_of_file = "/home/tim-external/Documents/icra2023FMS/figures/bunkerMapsDifferentTechniques/" + name_of_map

plt.savefig(name_of_file + ".pdf", bbox_inches='tight')
