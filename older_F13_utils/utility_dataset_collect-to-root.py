# Move files from subfolders to the root, assign randomized names

import os, random, string, shutil, sys

TARGET_FOLDER = "C:\\ThermalDataset"

def generate_random_letters(k=8):
    return ''.join(random.choices(string.ascii_uppercase, k=k))

def move_files_to_root(target_folder):
    subfolders = [f for f in os.listdir(target_folder)  if os.path.isdir(os.path.join(target_folder, f))]
    subfolders.sort()    
    for idx, subfolder in enumerate(subfolders, start=1):
        subfolder_path = os.path.join(target_folder, subfolder)
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                while True:
                    random_part = generate_random_letters(8)
                    new_filename = f"f{idx}_{random_part}{ext}"
                    new_file_path = os.path.join(target_folder, new_filename)
                    if not os.path.exists(new_file_path): break
                shutil.move(file_path, new_file_path)
                print(f"Moved '{file_path}' to '{new_file_path}'")

if __name__ == "__main__":
    target_folder = TARGET_FOLDER
    if len(sys.argv) >= 2: target_folder = sys.argv[1]
    if not os.path.isdir(target_folder):
        print(f"Error: '{target_folder}' is not a valid folder.")
        sys.exit(1)
    move_files_to_root(target_folder)
