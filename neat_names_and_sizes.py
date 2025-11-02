import os
import random
import string
from PIL import Image

# === CONFIG ===
INPUT_DIR = r"D:\\Datasorted"      # source folder
OUTPUT_DIR = r"D:\\hot_infantry"    # where processed images will be saved
TARGET_SIZE = (640, 640)  # desired bounding box

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allowed image extensions
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def random_prefix(n=10):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(n))


def format_filename(original):
    base = os.path.splitext(os.path.basename(original))[0]
    base = base[:32]  # crop to 32 chars
    prefix = random_prefix()
    return f"{prefix}_{base}.png"


def process_image(in_path, out_path):
    img = Image.open(in_path).convert("RGB")
    w, h = img.size

    # Rotate portrait images
    if h > w:
        img = img.rotate(90, expand=True)
        w, h = img.size

    # If image is bigger → shrink proportionally
    if w > TARGET_SIZE[0] or h > TARGET_SIZE[1]:
        img.thumbnail(TARGET_SIZE, Image.LANCZOS)
        w, h = img.size

    # If image is smaller → pad to 640x640 center
    if w < TARGET_SIZE[0] or h < TARGET_SIZE[1]:
        canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
        offset = ((TARGET_SIZE[0] - w) // 2, (TARGET_SIZE[1] - h) // 2)
        canvas.paste(img, offset)
        img = canvas

    img.save(out_path, "PNG")


# Walk through all files
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in EXTS:
            continue

        src = os.path.join(root, f)
        out = os.path.join(OUTPUT_DIR, format_filename(f))
        try:
            process_image(src, out)
            print(f"OK: {src} -> {out}")
        except Exception as e:
            print(f"FAILED: {src} | {e}")

print("Done")
