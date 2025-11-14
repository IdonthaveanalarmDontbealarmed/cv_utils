# Clusterize similar images using simple feature extraction and copy to an output folder.
# Useful if raw dataset with images could contain duplicates or very similar images which might create vision bias
# towards specific repersentation of object of interest. 
# Run > review output clusters > run again on 'unique' folder leftovers with adjusting treshold. 

import os
import shutil
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from PIL import Image

INPUT_FOLDER = "C:\\ThermalDataset\\extracted_drone_vid"
OUTPUT_FOLDER = "C:\\Users\\V\\Desktop\\Out"
SIMILARITY_THRESHOLD = 0.1  # Clustering sensitivity, use 0.3 - 0.7 for frames grabbed consequitively from a video, and less for individually captured images 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

image_paths = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
feature_vectors = []
print("Extracting features from images...")
for path in tqdm(image_paths): feature_vectors.append(extract_features(path))
feature_vectors = np.array(feature_vectors)

# Clustering using DBSCAN
print("Clustering images...")
clustering = DBSCAN(eps=SIMILARITY_THRESHOLD, min_samples=2, metric='euclidean').fit(feature_vectors)
labels = clustering.labels_

for idx, label in enumerate(labels):
    if label == -1: cluster_folder = os.path.join(OUTPUT_FOLDER, "unique")
    else: cluster_folder = os.path.join(OUTPUT_FOLDER, f"cluster_{label}")
    os.makedirs(cluster_folder, exist_ok=True)
    shutil.copy(image_paths[idx], cluster_folder)
print("Clustering complete. Images grouped in:", OUTPUT_FOLDER)
