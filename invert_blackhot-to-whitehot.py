import cv2
import os
from pathlib import Path

FOLDER = r"D:\\Datasorted\\Infusions\\Extra4"

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

for file in os.listdir(FOLDER):
    path = Path(FOLDER) / file
    if path.suffix.lower() not in EXT:
        continue
    
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Skipped {file} (can't read)")
        continue
    
    # invert (black-hot → white-hot)
    if img.dtype == "uint16":  # 16-bit thermal streams
        inverted = 65535 - img
    else:  # standard 8-bit exports
        inverted = 255 - img
    
    cv2.imwrite(str(path), inverted)
    print(f"✅ Inverted {file}")

print("🔥 All images inverted in place — black hot → white hot.")
