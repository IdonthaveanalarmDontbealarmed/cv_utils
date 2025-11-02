import cv2
import os
import shutil

IN_FOLDER = r"d:\in"
OUT_FOLDER = r"d:\out"
SIDE_FOLDER = r"d:\side"
AXERA_FOLDER = r"d:\axera"
# Make sure output folders exist
os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(SIDE_FOLDER, exist_ok=True)

# Supported image extensions
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

# Collect images
images = [f for f in os.listdir(IN_FOLDER) if os.path.splitext(f)[1].lower() in EXTENSIONS]
images.sort()

if not images:
    print("No images found in input folder")
    exit()

index = 0

def show_image(idx):
    path = os.path.join(IN_FOLDER, images[idx])
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot open {path}")
        return
    cv2.imshow("Image Browser", img)
    cv2.setWindowTitle("Image Browser", f"{images[idx]} ({idx+1}/{len(images)})")

show_image(index)

while True:
    key = cv2.waitKey(0)

    # Next image (Right arrow or 'd')
    if key in [ord('d'), 2555904]:
        index = (index + 1) % len(images)
        show_image(index)

    # Previous image (Left arrow or 'a')
    elif key in [ord('a'), 2424832]:
        index = (index - 1) % len(images)
        show_image(index)

    # SPACE — copy to OUT_FOLDER
    elif key == 32:
        src = os.path.join(IN_FOLDER, images[index])
        dst = os.path.join(OUT_FOLDER, images[index])
        shutil.copy2(src, dst)
        print(f"[OUT] Copied: {images[index]}")
        index = (index + 1) % len(images)
        show_image(index)

    # S — copy to SIDE_FOLDER
    elif key in [ord('s'), ord('S')]:
        src = os.path.join(IN_FOLDER, images[index])
        dst = os.path.join(SIDE_FOLDER, images[index])
        shutil.copy2(src, dst)
        print(f"[SIDE] Copied: {images[index]}")
        index = (index + 1) % len(images)
        show_image(index)

    # F — copy to AXERA_FOLDER
    elif key in [ord('f'), ord('F')]:
        src = os.path.join(IN_FOLDER, images[index])
        dst = os.path.join(AXERA_FOLDER, images[index])
        shutil.copy2(src, dst)
        print(f"[SIDE] Copied: {images[index]}")
        index = (index + 1) % len(images)
        show_image(index)

    # ESC — exit
    elif key == 27:
        break

cv2.destroyAllWindows()
