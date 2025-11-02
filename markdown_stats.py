import os
import cv2
from collections import defaultdict

# --- CONFIG ---
INPUT_FOLDER = r"D:\\hot_infantry_crop"  # change to your folder
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- DATA STRUCTS ---
class_counts = defaultdict(int)
class_pixel_widths = defaultdict(float)
class_pixel_heights = defaultdict(float)
class_bbox_count = defaultdict(int)

annotated_images = 0

for file in os.listdir(INPUT_FOLDER):
    if not file.endswith(".txt"):
        continue

    label_path = os.path.join(INPUT_FOLDER, file)
    img_name = os.path.splitext(file)[0]

    # find matching image
    img_file = None
    for ext in IMG_EXTS:
        candidate = os.path.join(INPUT_FOLDER, img_name + ext)
        if os.path.exists(candidate):
            img_file = candidate
            break

    if img_file is None:
        continue  # skip orphan labels

    # load image to get dimensions
    img = cv2.imread(img_file)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    if not lines:
        continue

    annotated_images += 1

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls = int(parts[0])
        _, x_center, y_center, bb_w, bb_h = map(float, parts)

        # convert normalized to pixels
        box_w_px = bb_w * w
        box_h_px = bb_h * h

        class_counts[cls] += 1
        class_pixel_widths[cls] += box_w_px
        class_pixel_heights[cls] += box_h_px
        class_bbox_count[cls] += 1

# --- RESULTS ---
print(f"Annotated images: {annotated_images}")
print()

for cls in sorted(class_counts.keys()):
    total = class_counts[cls]
    avg_per_img = total / annotated_images if annotated_images > 0 else 0

    avg_w = class_pixel_widths[cls] / class_bbox_count[cls]
    avg_h = class_pixel_heights[cls] / class_bbox_count[cls]

    print(f"Class {cls}:")
    print(f"  Boxes: {total}")
    print(f"  Avg boxes per image: {avg_per_img:.3f}")
    print(f"  Avg box size: {avg_w:.1f}px × {avg_h:.1f}px")
    print()
