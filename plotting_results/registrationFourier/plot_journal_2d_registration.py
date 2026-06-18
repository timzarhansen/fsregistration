import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# THIS FILE WORKS ONLY WITH PLOT ALL SOLUTIONS
ordner_save = "/home/ws/matlab/registrationFourier/resultsJournalFMS2D/pdffigures/"
name_of_batch_file = "batchfile.sh"

resampled_data_for_sphere1 = np.genfromtxt("csvFiles/resampledVoxel1.csv", delimiter=",")

size_of_spheres = int(np.sqrt(resampled_data_for_sphere1.shape[0]))

resampled_data_for_sphere_result1 = np.zeros((size_of_spheres, size_of_spheres))

for j in range(size_of_spheres):
    for i in range(size_of_spheres):
        resampled_data_for_sphere_result1[j, i] = resampled_data_for_sphere1[(i - 1) * size_of_spheres + j]

# currently not that interesting
if True:
    fig = plt.figure(4)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1000 * np.outer(np.cos(u), np.sin(v))
    y = 1000 * np.outer(np.sin(u), np.sin(v))
    z = 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=cm.ScalarMappable(cmap=cm.viridis).to_rgba(resampled_data_for_sphere_result1), edgecolor='none')
    plt.axis('equal')
    fig.canvas.manager.set_window_title('Figure 4')
    fig.set_size_inches(600/100, 570/100, forward=True)
else:
    fig = plt.figure(4)
    plt.clf()
    plt.imshow(resampled_data_for_sphere_result1, aspect='equal')
    fig.set_size_inches(600/100, 570/100, forward=True)

name_of_map = "enhancedWrappedSphere"
name_of_file = ordner_save + name_of_map

plt.rcParams['text.usetex'] = True

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

# systemCommand = "pdfcrop " + nameOfFile +".pdf "+ nameOfFile+".pdf "
# os.system(systemCommand)

fig = plt.figure(6)
plt.clf()
correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")
x_for_plot = np.arange(0, len(correlation_of_angles))
x_for_plot = x_for_plot / len(correlation_of_angles) * 360
plt.plot(x_for_plot, correlation_of_angles)

fig.set_size_inches(400/100, 400/100, forward=True)
plt.grid(True)

plt.ylabel("Correlation Value")
plt.xlabel("angle in degree")

name_of_map = "1DCorrelation"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

# resultSize = sizeOfSpheres

fig = plt.figure(8)

plt.clf()
x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, correlation_matrix_shift_2d, edgecolor='none')
plt.xlabel("x-axis")
plt.ylabel("y-axis")

name_of_map = "2DCorrelationWithCorrection"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")

number_of_solutions_overall = 0
for i in range(int(data_information[0])):
    number_of_solutions_overall += data_information[i + 1]
cell_size = data_information[data_information[0] + 2]

fig = plt.figure(8)
plt.clf()
x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, correlation_matrix_shift_2d, edgecolor='none')
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.hold(True)
for current_translation_number in range(1, int(data_information[0 + 2]) + 1):
    potential_transformation_read_in = np.genfromtxt("csvFiles/potentialTransformation" + str(current_translation_number - 1) + ".csv", delimiter=",")
    transformation_t_mpsi = potential_transformation_read_in[0:4, 0:4]
    potential_transformation_read_in[0, 3] = potential_transformation_read_in[0, 3] / cell_size
    potential_transformation_read_in[1, 3] = potential_transformation_read_in[1, 3] / cell_size

    x_of_interest = -potential_transformation_read_in[0, 3] + np.ceil(result_size / 2)
    y_of_interest = -potential_transformation_read_in[1, 3] + np.ceil(result_size / 2)

    z_for_plot = correlation_matrix_shift_2d[int(round(y_of_interest)), int(round(x_of_interest))]
    ax.plot3D([x_of_interest], [y_of_interest], [z_for_plot], 'r+')

name_of_map = "2DCorrelationPeakDetectionFew"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(11)
plt.clf()

plt.imshow(magnitude1, aspect='equal')

name_of_map = "magnitude1"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(22)
plt.clf()

plt.imshow(magnitude2, aspect='equal')

name_of_map = "magnitude2"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(1)
plt.clf()

plt.title('Voxel: ' + str(1))

plt.imshow(voxel_data1, aspect='equal')

name_of_map = "voxelData1"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(2)
plt.clf()

plt.imshow(voxel_data2, aspect='equal')

name_of_map = "voxelData2"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(14)

ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1000 * np.outer(np.cos(u), np.sin(v))
y = 1000 * np.outer(np.sin(u), np.sin(v))
z = 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, facecolors=cm.ScalarMappable(cmap=cm.viridis).to_rgba(resampled_data_for_sphere_result1), edgecolor='none')
plt.axis('equal')

name_of_map = "2DSphere1"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(15)

ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1000 * np.outer(np.cos(u), np.sin(v))
y = 1000 * np.outer(np.sin(u), np.sin(v))
z = 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, facecolors=cm.ScalarMappable(cmap=cm.viridis).to_rgba(resampled_data_for_sphere_result2), edgecolor='none')
plt.axis('equal')

name_of_map = "2DSphere2"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(16)
plt.imshow(resampled_data_for_sphere_result1, aspect='equal')

name_of_map = "2DUnwrappedSphere1"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)

fig = plt.figure(17)
plt.imshow(resampled_data_for_sphere_result2, aspect='equal')

name_of_map = "2DUnwrappedSphere2"
name_of_file = ordner_save + name_of_map

plt.savefig(name_of_file, format='pdf')
add_command_to_batchfile(name_of_batch_file, 'pdfcrop ' + name_of_file + '.pdf ' + name_of_file + '.pdf &', True)
