import os
import sys
from pillow_heif import open_heif
from PIL import Image

DEFAULT_FOLDER = "C:\\Users\\V\\Desktop\\stark"

def convert_heic_to_png(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    heic_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".heic")]
    if not heic_files:
        print("No HEIC images found in the folder.")
        return
    
    for heic_file in heic_files:
        heic_path = os.path.join(folder_path, heic_file)
        png_path = os.path.join(folder_path, os.path.splitext(heic_file)[0] + ".png")
        
        if os.path.exists(png_path):
            print(f"Skipping {heic_file}, PNG already exists.")
            continue
        
        try:
            heif_image = open_heif(heic_path)
            image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
            image.save(png_path, format="PNG")
            print(f"Converted {heic_file} -> {png_path}")
        except Exception as e:
            print(f"Failed to convert {heic_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        convert_heic_to_png(DEFAULT_FOLDER)
    else:
        convert_heic_to_png(sys.argv[1])
