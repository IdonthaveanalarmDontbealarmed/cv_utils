# Clusterize similar images using hotspot extraction > feature comparison and copy to an output folder.
# Useful if raw dataset with images could contain visual "duplicates" to ease training and protect from 
# creating vision bias towards specific look of an object of interest. For huge datasets with few classes
# and where the objects of interest have multiple real-life versions, dobiusly distinctive for CV. 
# Run > review output clusters > run again on 'unique' folder leftovers with adjusting treshold. 

import os, shutil, cv2, numpy as np, torch
import torchvision.models as models, torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from PIL import Image

INPUT_FOLDER = "C:\\ThermalDataset\\extracted_drone_vid"
OUTPUT_FOLDER = "C:\\Users\\V\\Desktop\\Out"
SIMILARITY_THRESHOLD = 0.3
MIN_SAMPLES = 2
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(DEVICE)
model.eval()
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def extract_hotspots(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return []
        blurred = cv2.GaussianBlur(image, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots = [image[y:y+h, x:x+w] for x,y,w,h in (cv2.boundingRect(c) for c in contours) if w>=10 and h>=10]
        return hotspots if hotspots else [image]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def extract_features_from_hotspots(image_path):
    hotspots = extract_hotspots(image_path)
    feature_vectors = []
    for hotspot in hotspots:
        try:
            hotspot_resized = cv2.resize(hotspot, (224,224))
            hotspot_rgb = cv2.cvtColor(hotspot_resized, cv2.COLOR_GRAY2RGB)
            input_tensor = transform(Image.fromarray(hotspot_rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                features = model(input_tensor)
            feature_vectors.append(features.cpu().numpy().flatten())
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
    return np.mean(feature_vectors, axis=0) if feature_vectors else np.zeros(2048)

image_paths = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('png','jpg','jpeg'))]
features_list = [extract_features_from_hotspots(path) for path in tqdm(image_paths, desc="Extracting features")]
features_array = np.array(features_list)
clustering = DBSCAN(eps=SIMILARITY_THRESHOLD, min_samples=MIN_SAMPLES, metric='euclidean').fit(features_array)
labels = clustering.labels_
for idx, label in enumerate(labels):
    folder = os.path.join(OUTPUT_FOLDER, "unique" if label == -1 else f"cluster_{label}")
    os.makedirs(folder, exist_ok=True)
    try:
        shutil.copy(image_paths[idx], folder)
    except Exception as e:
        print(f"Error copying {image_paths[idx]}: {e}")
print("Clustering complete. Images grouped in:", OUTPUT_FOLDER)
