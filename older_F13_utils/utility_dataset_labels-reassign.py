# Batch-edit labels in txt files

import os

LABELS_FOLDER = 'C:\\ver3\\val\\labels'

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            if parts[0] == '2':  # Reassign label 2 to label 0
                parts[0] = '0'
                file.write(' '.join(parts) + '\n')
            else: pass  # Remove other labels

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)

process_folder(LABELS_FOLDER)
