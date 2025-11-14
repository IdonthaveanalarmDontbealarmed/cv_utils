# This utility will copy marked down images and labels from an input to an output file folder,
# with checking consistency and creating YOLO-style folder structures. It might be useful in case
# if [LabelStudio (https://labelstud.io/)], [YoloLabel (https://github.com/developer0hye/Yolo_Label)] 
# or a similar GUI tool is used for marking images down, and all files are put together in one folder. 

import os, sys, shutil, random
from PIL import Image

INPUT_DIR = r"C:\\python\\hot-infantry-dataset"
OUTPUT_DIR = r"C:\\python\\datasets\\ippt"
HIGHLIGHTS_LIST_FILE = r"C:\\Users\\V\\Desktop\\empty.txt"  # Optionally, set aside files using list of filenames separated by line break
TRAIN_RATIO = 0.88  # Set aside (1 - VALUE)% of images for validation
IMG_FORMAT = ".png"  # Image files extension
LAB_FORMAT = ".txt"  # Label files extension

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
elif os.listdir(OUTPUT_DIR):
    print("Output folder exists and is not empty. Halting.")
    sys.exit(1)

if not os.path.exists(HIGHLIGHTS_LIST_FILE):
    print("Highlights list file not found. Halting.")
    sys.exit(1)
with open(HIGHLIGHTS_LIST_FILE, "r") as f:
    hl_list = [l.strip() for l in f if l.strip()]
for hl in hl_list:
    if not os.path.exists(os.path.join(INPUT_DIR, hl)):
        print(f"Highlight file {hl} not found in input folder. Halting.")
        sys.exit(1)

for sub in ["train", "val", "highlights"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub, "images"))
    os.makedirs(os.path.join(OUTPUT_DIR, sub, "labels"))
os.makedirs(os.path.join(OUTPUT_DIR, "broken"))

def valid_label(path):
    try:
        with open(path, "r") as f: lines = f.readlines()
        if not lines: return False
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: return False
            int(parts[0])
            for p in parts[1:]: float(p)
        return True
    except: return False

def valid_image(path):
    try:
        with Image.open(path) as im: im.verify()
        return True
    except: return False

def copy_pair(dest, img, lab):
    shutil.copy2(img, os.path.join(dest, "images", os.path.basename(img)))
    shutil.copy2(lab, os.path.join(dest, "labels", os.path.basename(lab)))

def copy_broken(*files):
    for f in files:
        if f and os.path.exists(f): shutil.copy2(f, os.path.join(OUTPUT_DIR, "broken", os.path.basename(f)))

processed = set()
for fname in os.listdir(INPUT_DIR):
    fpath = os.path.join(INPUT_DIR, fname)
    if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() == IMG_FORMAT:
        processed.add(fname)
        labname = os.path.splitext(fname)[0] + LAB_FORMAT
        labpath = os.path.join(INPUT_DIR, labname)
        ok = True
        if not os.path.exists(labpath):
            print(f"Label missing for {fname}.")
            ok = False
        if not valid_image(fpath):
            print(f"Image {fname} is broken.")
            ok = False
        if ok and not valid_label(labpath):
            print(f"Label file {labname} invalid.")
            ok = False
        if not ok:
            print(f"Copying {fname} (and label if exists) to broken.")
            copy_broken(fpath, labpath if os.path.exists(labpath) else None)
        else:
            if fname in hl_list:
                print(f"Copying {fname} to highlights.")
                copy_pair(os.path.join(OUTPUT_DIR, "highlights"), fpath, labpath)
            else:
                dest = os.path.join(OUTPUT_DIR, "train") if random.random() < TRAIN_RATIO else os.path.join(OUTPUT_DIR, "val")
                print(f"Copying {fname} to {os.path.basename(dest)}.")
                copy_pair(dest, fpath, labpath)
for fname in os.listdir(INPUT_DIR): 
    fpath = os.path.join(INPUT_DIR, fname)
    if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() == LAB_FORMAT:
        imgname = os.path.splitext(fname)[0] + IMG_FORMAT
        if imgname not in processed:
            print(f"Label file {fname} without corresponding image. Marking as broken.")
            copy_broken(fpath)
for fname in os.listdir(INPUT_DIR):
    fpath = os.path.join(INPUT_DIR, fname)
    if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() not in [LAB_FORMAT, IMG_FORMAT]:
        print(f"Irrelevant file {fname}. Copying to broken.")
        copy_broken(fpath)
print("Processing complete.")