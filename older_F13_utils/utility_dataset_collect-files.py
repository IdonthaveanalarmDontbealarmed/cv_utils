# Copy images from an input to an output folder with ensuring files are not overwritten.
# Review images on the go, ENTER > copy, SPACE > skip.
# Useful when compiling data from multiple folders, memory cards, etc.

import os
import cv2

INPUT_FOLDER = "C:\\ThermalDataset\\extracted_drone_human_vid_2"
OUTPUT_FOLDER = "C:\\ThermalDataset\\"

def get_unique_filename(folder, filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename    
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename

def process_images():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    total_images = len(images)    
    cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)
    for index, img_name in enumerate(images, start=1):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        upscaled_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        cv2.resizeWindow("Image Viewer", upscaled_img.shape[1], upscaled_img.shape[0])
        cv2.setWindowTitle("Image Viewer", f"Image Viewer ({index}/{total_images}): {img_name}")
        cv2.imshow("Image Viewer", upscaled_img)
        key = cv2.waitKey(0)
        if key == 13:  # Enter key - copy
            new_name = get_unique_filename(OUTPUT_FOLDER, img_name)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, new_name), img)
        elif key == 32: pass  # Space key - skip
    cv2.destroyAllWindows()

if __name__ == "__main__": process_images()
