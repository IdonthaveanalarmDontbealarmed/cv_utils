import cv2
import numpy as np
import random, string, os, time

ZOOM_RANGE = (2000, 16000)
BR_RANGE = (0, 100)
HUE_RANGE = (0, 7)
QUALITY_RANGE = (0, 2)

PROPERTY_ALIASES = [None] * 100
PROPERTY_ALIASES[3] = "Cap Width"
PROPERTY_ALIASES[4] = "Cap Height"
PROPERTY_ALIASES[10] = "Brightness"
PROPERTY_ALIASES[11] = "Contrast"
PROPERTY_ALIASES[12] = "[1-3] Quality"
PROPERTY_ALIASES[13] = "[A-D] Tint"
PROPERTY_ALIASES[17] = "[G] Hide GUI"
PROPERTY_ALIASES[27] = "[W-S] Zoom"
PROPERTY_ALIASES[37] = "[E] UVC GUI"

def list_uvc_cameras():
    devices = []
    print("\nScanning for UVC cameras...")
    for index in range(10):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                devices.append(index)
                cap.release()
            else: break
        except: pass
    if not devices: print("No UVC cameras detected.")
    else: print(f"Available UVC Cameras: {devices}")
    return devices

def get_camera_info(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("\n--- Camera Info ---")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print("---------------------")

def list_camera_controls(cap):
    properties = {}
    print("\n--- Detected Camera Properties ---")
    for i in range(100):
        value = cap.get(i)
        if value != -1:
            alias = PROPERTY_ALIASES[i] if PROPERTY_ALIASES[i] else f"Property {i}"
            print(f"{alias}: {value}")
            properties[i] = value
    print("--------------------------------")
    return properties

def set_initial_camera_settings(cap):
    pass
    # cap.set(12, QUALITY_RANGE[-1])
    # cap.set(27, ZOOM_RANGE[0])

def toggle_zoom(cap, val):
    prop = 27  
    zoom_value = int(cap.get(prop))
    new_zoom = np.clip(zoom_value + val, ZOOM_RANGE[0], ZOOM_RANGE[1])
    cap.set(prop, new_zoom)

def adjust_tint(cap, val):
    prop = 13  
    value = int(cap.get(prop))
    new_value = np.clip(value + val, HUE_RANGE[0], HUE_RANGE[1])
    cap.set(prop, new_value)

def adjust_brightness(cap, val):
    prop = 10  
    value = int(cap.get(prop))
    new_value = np.clip(value + val, BR_RANGE[0], BR_RANGE[1])
    cap.set(prop, new_value)

def set_quality(cap, level):
    prop = 12  
    if level in [0, 1, 2]:
        cap.set(prop, level)

def print_keyboard_mapping():
    print("\n--- Keyboard Controls ---")
    print("W/S - Adjust Zoom (1000-16000)")
    print("A/D - Adjust Tint (0-7)")
    print("Z/C - Adjust Brightness (0-100)")
    print("1/2/3 - Set Quality (0-2)")
    print("E - Open Camera Settings GUI")
    print("Q - Quit")
    print("--------------------------")

def draw_properties_on_frame(frame, properties):
    height, width, _ = frame.shape
    overlay_width = width // 2
    extended_frame = np.zeros((height, width + overlay_width, 3), dtype=np.uint8)
    extended_frame[:, :width] = frame
    y_offset = 30
    for prop, value in properties.items():
        alias = PROPERTY_ALIASES[prop] if PROPERTY_ALIASES[prop] else f"Property {prop}"
        text = f"{alias}: {value:.1f}"
        color = (29, 147, 248) if PROPERTY_ALIASES[prop] is not None else (212, 212, 212)
        cv2.putText(extended_frame, text, (width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y_offset += 25
    cv2.putText(extended_frame, "[Q] Quit", (width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (29, 147, 248), 1, cv2.LINE_AA)
    return extended_frame

def rand_id(n=8): 
    return "".join(random.choice(string.ascii_uppercase) for _ in range(n))

def capture_with_opencv():
    devices = list_uvc_cameras()
    if not devices:
        print("No UVC cameras found. Exiting.")
        return
    camera_index = devices[-1]
    camera_index = 2
    print(f"\nUsing camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    set_initial_camera_settings(cap)
    get_camera_info(cap)
    properties = list_camera_controls(cap)
    print_keyboard_mapping()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame grab failed.")
            break
        properties = {k: cap.get(k) for k in properties.keys()}
        extended_frame = draw_properties_on_frame(frame, properties)
        cv2.imshow("Thermal Camera (OpenCV)", extended_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("w"): toggle_zoom(cap, +1000)
        elif key == ord("s"): toggle_zoom(cap, -1000)
        elif key == ord("a"): adjust_tint(cap, -1)
        elif key == ord("d"): adjust_tint(cap, +1)
        elif key == ord("z"): adjust_brightness(cap, -5)
        elif key == ord("c"): adjust_brightness(cap, +5)
        elif key == ord("e"): cap.set(37, 1)
        elif key == ord("g"): cap.set(17, 1)
        elif key == ord("1"): set_quality(cap, 0)
        elif key == ord("2"): set_quality(cap, 1)
        elif key == ord("3"): set_quality(cap, 2)
        elif key==32:
            uid=rand_id()
            cv2.imwrite(f"NEWCAP/cap_{uid}_t.jpg",frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": capture_with_opencv()
