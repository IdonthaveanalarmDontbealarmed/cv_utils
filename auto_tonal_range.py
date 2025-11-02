import cv2
import os
import numpy as np

IN_FOLDER = r"D:\\hot_infantry_crop"
OUT_FOLDER = r"D:\\hot_infantry_crop_autotone"
os.makedirs(OUT_FOLDER, exist_ok=True)

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def stretch_channel(ch):
    min_val = np.min(ch)
    max_val = np.max(ch)
    if max_val == min_val:  # Avoid divide-by-zero
        return ch
    return ((ch - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)

for fname in os.listdir(IN_FOLDER):
    if os.path.splitext(fname)[1].lower() not in EXT:
        continue

    img = cv2.imread(os.path.join(IN_FOLDER, fname))

    if img is None:
        print("Skipping unreadable:", fname)
        continue

    # Split, scale each channel, merge back
    channels = cv2.split(img)
    stretched = [stretch_channel(c) for c in channels]
    img_out = cv2.merge(stretched)

    cv2.imwrite(os.path.join(OUT_FOLDER, fname), img_out)
    print("Processed:", fname)

print("Done ✅")
