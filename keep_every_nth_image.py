import os
from pathlib import Path
import shutil

# === CONFIG ===
folder = r"D:\\Datasorted\\Infusions\\Patrol"   # your folder with images
N = 5                            # keep every Nth frame
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# === SAFETY: optional backup folder ===
backup = None  # e.g., r"D:\BACKUP" — or set to None to skip backup

p = Path(folder)
files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])

if not files:
    print("No images found.")
    exit()

print(f"Found {len(files)} images. Keeping every {N}th...")

kept = 0
removed = 0

for idx, file in enumerate(files):
    if idx % N == 0:
        kept += 1
        continue

    # backup if needed
    if backup:
        dest = Path(backup) / file.name
        shutil.move(str(file), dest)
    else:
        file.unlink()  # delete

    removed += 1

print(f"✅ Done! Kept: {kept}   ❌ Removed: {removed}")
