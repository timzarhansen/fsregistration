import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

WHICH_METHOD_USED = 6

for index in range(3, 55):
    map_data = pd.read_csv(f"/home/tim-external/dataFolder/SimulationEnvironment/experimentForVideoICRA/scanNumber{index}.csv", header=None).values
    
    N = 256
    voxel_data = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            voxel_data[k, j] = map_data[(k - 1) * N + j]
    
    map_data = voxel_data
    
    N = int(np.sqrt(map_data.shape[0]))
    map_data = np.flip(map_data, axis=1)
    map_data = np.rot90(np.rot90(np.rot90(map_data)))
    
    fig = plt.figure(4, figsize=(8, 8))
    plt.imshow(map_data)
    plt.ylabel("120 m")
    plt.xlabel("120 m")
    plt.box(on=True)
    plt.pause(0.5)
    plt.close(fig)
