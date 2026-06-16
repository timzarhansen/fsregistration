import os
import pandas as pd

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

folder_path = 'resultsAnt'  # Replace if needed

files = os.listdir(folder_path)

# Filter out directories
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

for filename in files:
    file_path = os.path.join(folder_path, filename)
    try:
        data = pd.read_csv(file_path)
        dataset_size = len(data)
        print(f'File: {file_path}, Dataset Size: {dataset_size}')
    except Exception as e:
        print(f'Error reading file: {file_path} - {str(e)}')
