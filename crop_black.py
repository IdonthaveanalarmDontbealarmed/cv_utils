import cv2
import os
from pathlib import Path
import numpy as np

INPUT_DIR = r"D:\\hot_infantry"
OUTPUT_DIR = r"D:\\hot_infantry_crop"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allowed extensions
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask of non-black pixels
    mask = gray > 0  

    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return img[y_min:y_max+1, x_min:x_max+1]

for file in os.listdir(INPUT_DIR):
    ext = Path(file).suffix.lower()
    if ext not in exts:
        continue

    img_path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Can't open {file}")
        continue

    cropped = crop_black_borders(img)
    out_path = os.path.join(OUTPUT_DIR, file)
    cv2.imwrite(out_path, cropped)
    print(f"✅ Cropped: {file}")
