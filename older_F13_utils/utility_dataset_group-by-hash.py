# Clusterize similar images using hash extraction > comparison and copy to an output folder.
# Useful if raw dataset with images has plenty of visual "duplicates", for instance consecutive frames
# taken from the same video stream - to ease training and protect from creating vision bias towards 
# specific look of an object of interest. For huge datasets with few classes and where the objects 
# of interest have multiple real-life versions, dobiusly distinctive for CV. 

import os, shutil
import imagehash
from PIL import Image
from tqdm import tqdm

INPUT_FOLDER = "C:\\ThermalDataset"
OUTPUT_FOLDER = "C:\\Users\\V\\Desktop\\Out"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHUNK_SIZE = 500  # How many images to process at one bite
DIST_THRESHOLD = 5  # Hamming distance threshold for pHash near-duplicates
DOWNSAMPLE_SIZE = (320, 256)  # Downsample for faster hashing

def compute_phash(img_path, size=(256, 256)):
    try:
        with Image.open(img_path) as img:
            # Downsample for speed; only used for hashing
            img = img.convert('L')  # Convert to grayscale if desired
            img.thumbnail(size)     # In-place downsample
            return imagehash.phash(img)
    except:
        return None

def cluster_chunk(chunk_phashes, threshold):
    n = len(chunk_phashes)
    visited = [False]*n
    clusters = []
    cluster_reps = []
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        if chunk_phashes[i] is None: continue
        for j in range(i+1, n):
            if chunk_phashes[j] is None: continue
            dist = chunk_phashes[i] - chunk_phashes[j]
            if dist <= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    for i in range(n):  
        if visited[i] or chunk_phashes[i] is None: continue
        queue = [i]
        visited[i] = True
        cluster = [i]
        while queue:
            curr = queue.pop()
            for neighbor in adjacency[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    cluster.append(neighbor)
        clusters.append(cluster)
        cluster_reps.append(chunk_phashes[cluster[0]])
    return clusters, cluster_reps

def merge_clusters(global_clusters, global_reps, chunk_clusters, chunk_reps, image_paths, threshold):
    for i, local_rep in enumerate(chunk_reps):
        best_cluster = -1
        best_dist = 999999
        for g_idx, g_rep in enumerate(global_reps):
            dist = g_rep - local_rep
            if dist < best_dist:
                best_dist = dist
                best_cluster = g_idx
        if best_dist <= threshold: global_clusters[best_cluster].extend(chunk_clusters[i])
        else:
            global_clusters.append(chunk_clusters[i])
            global_reps.append(local_rep)

image_paths = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(('png','jpg','jpeg'))
]
global_clusters = [] 
global_reps = []
print("Processing images in chunks...")
for start_idx in range(0, len(image_paths), CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, len(image_paths))
    chunk_paths = image_paths[start_idx:end_idx]
    chunk_phashes = []
    for p in tqdm(chunk_paths, desc=f"Chunk {start_idx}-{end_idx}"):
        ph = compute_phash(p, size=DOWNSAMPLE_SIZE)
        chunk_phashes.append(ph)
    chunk_clusters, chunk_reps = cluster_chunk(chunk_phashes, DIST_THRESHOLD)
    chunk_global_indices = list(range(start_idx, end_idx))
    mapped_clusters = []
    for c in chunk_clusters:
        mapped = [chunk_global_indices[idx] for idx in c]
        mapped_clusters.append(mapped)
    merge_clusters(global_clusters, global_reps, mapped_clusters, chunk_reps, chunk_paths, DIST_THRESHOLD)

print("Organizing final clusters...")
for cluster_id, cluster_members in enumerate(global_clusters):
    cluster_folder = os.path.join(OUTPUT_FOLDER, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)
    for member_idx in cluster_members:
        src_path = image_paths[member_idx]
        try: shutil.copy(src_path, cluster_folder)
        except Exception as e: print(f"Error copying {src_path}: {e}")
print("Done! Clusters have been created in:", OUTPUT_FOLDER)
