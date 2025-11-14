# Redistribute files from a target folder into its new subfolders with a given maximum number of items in each 

import os, random, string, shutil, sys

TARGET_FOLDER = "C:\\ThermalDataset\\extracted_drone_coarse"
DEFAULT_FILES_PER_FOLDER = 500

def generate_random_letters(k=8):
    return ''.join(random.choices(string.ascii_uppercase, k=k))

def break_files_into_subfolders(target_folder, files_per_folder):
    files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    files.sort()
    total_files = len(files)
    folder_index = 1

    for i in range(0, total_files, files_per_folder):
        subfolder_path = os.path.join(target_folder, f"f{folder_index}")
        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder '{subfolder_path}'")
        chunk = files[i:i+files_per_folder]
        for filename in chunk:
            source_file = os.path.join(target_folder, filename)
            destination_file = os.path.join(subfolder_path, filename)
            if os.path.exists(destination_file):
                base, ext = os.path.splitext(filename)
                while True:
                    random_part = generate_random_letters(8)
                    new_filename = f"{base}_{random_part}{ext}"
                    destination_file = os.path.join(subfolder_path, new_filename)
                    if not os.path.exists(destination_file): break
            shutil.move(source_file, destination_file)
            print(f"Moved '{source_file}' to '{destination_file}'")
        folder_index += 1

if __name__ == "__main__":
    target_folder = TARGET_FOLDER
    files_per_folder = DEFAULT_FILES_PER_FOLDER
    if len(sys.argv) >= 2: target_folder = sys.argv[1]
    if len(sys.argv) >= 3:
        try: files_per_folder = int(sys.argv[2])
        except ValueError:
            print("Error: number of files per folder must be an integer.")
            sys.exit(1)
    if not os.path.isdir(target_folder):
        print(f"Error: '{target_folder}' is not a valid folder.")
        sys.exit(1)
    break_files_into_subfolders(target_folder, files_per_folder)
