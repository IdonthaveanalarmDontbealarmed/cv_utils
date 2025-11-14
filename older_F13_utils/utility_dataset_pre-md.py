from ultralytics import YOLO
from pathlib import Path
import cv2

# paths
model_path = Path("C:\\python\\game-of-drones\\runs\\detect\\train5\\weights\\best.pt")
images_dir = Path("C:\\python\\hot-infantry-dataset")

# load model
model = YOLO(str(model_path))

# iterate images
for img_path in images_dir.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]: 
        continue
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        continue
    try:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        results = model(im)
        boxes = results[0].boxes
        lines = []
        for b in boxes:
            cls = int(b.cls[0])
            x, y, w, h = b.xywhn[0].tolist()
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        if lines:
            txt_path.write_text("\n".join(lines))
    except Exception:
        continue