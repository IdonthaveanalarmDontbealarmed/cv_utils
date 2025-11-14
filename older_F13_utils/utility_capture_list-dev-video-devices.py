import os, cv2

print("Enumerating /dev/video* devices...")
os.system("v4l2-ctl --list-devices")

print("\nTrying to open cameras via OpenCV:")
for i in range(10):
    path = f"/dev/video{i}"
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        print(f"Camera found at {path}")
        cap.release()
    else:
        print(f"No camera at {path}")
