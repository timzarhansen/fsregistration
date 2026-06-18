import numpy as np
import matplotlib.pyplot as plt
import os


# clc, clear - skipped

# Specify your BASE FOLDER path here (containing all test folders)
# base_path = '/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImages/FirstTestResults/64'
# base_path = '/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImages/FirstTestResults/128'
# base_path = '/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImages/FirstTestResults/256'
base_path = '/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImages/FirstTestResults/512'

# Get all immediate subfolders
test_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

rot_data = []
trans_data = []
folder_labels = []

for folder_info in test_folders:
    # Skip system folders
    if folder_info in ['.', '..']:
        continue

    # Get full paths to potential CSV files
    rot_path = os.path.join(base_path, folder_info, 'rotationErrorList.csv')
    trans_path = os.path.join(base_path, folder_info, 'translationErrorList.csv')

    # Check if both files exist before processing
    if not os.path.exists(rot_path) or not os.path.exists(trans_path):
        continue

    # Read rotation errors (degrees)
    rot_errors = np.genfromtxt(rot_path)
    rot_data.append(rot_errors.flatten())

    # Special parsing for translation data with brackets and spaces
    try:
        with open(trans_path, 'r') as f:
            lines = f.readlines()
            trans_data_list = [list(map(float, line.strip().split())) for line in lines if line.strip()]
            nx2_matrix = np.array(trans_data_list)

            if len(nx2_matrix) > 0:
                # Calculate magnitude of translation errors
                trans_mag = np.sqrt(np.sum(nx2_matrix ** 2, axis=1))
                trans_data.append(trans_mag)
            else:
                continue
    except Exception as e:
        print(f'Error processing {trans_path}: {e}')
        continue

    folder_labels.append(folder_info)

# Find the maximum length of any group in rot_data
max_length = max([len(vec) for vec in rot_data])

# Initialize a matrix filled with NaNs
X = np.full((max_length, len(rot_data)), np.nan)

# Fill each column with your data
for i, vec in enumerate(rot_data):
    X[:len(vec), i] = vec

# Plot using the numeric matrix
plt.figure()
plt.boxplot(X, labels=folder_labels)
plt.title('Rotation Errors (Degrees)')
plt.xlabel('Test Folder')
plt.ylabel('Error [°]')

# Find the maximum length of any group in trans_data
max_length = max([len(vec) for vec in trans_data])

# Initialize a matrix filled with NaNs
X = np.full((max_length, len(trans_data)), np.nan)

# Fill each column with your data
for i, vec in enumerate(trans_data):
    X[:len(vec), i] = vec

# Plot using the numeric matrix
plt.figure()
plt.boxplot(X, labels=folder_labels)
plt.title('Translation Errors Magnitude')
plt.xlabel('Test Folder')
plt.ylabel('Error [units]')
