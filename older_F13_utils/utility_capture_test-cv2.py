import os, cv2, numpy as np, random, string, sys

CAM_INDEX=1
REQ_W, REQ_H = 640, 480
UPSCALE=2
SAVE_PREFIX="kh_"

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
if not cap.isOpened(): print("Failed to open camera..."); raise SystemExit

# Ask for 640x480 (driver may override)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)

# Make sure the output folder exists
os.makedirs("CAP", exist_ok=True)

def new_path(prefix=SAVE_PREFIX, n=8, ext=".png"):
    while True:
        name = prefix + "".join(random.choices(string.ascii_uppercase, k=n)) + ext
        p = os.path.join("CAP", name)
        if not os.path.exists(p): return p

def stretch_u16_to_u8(img16, p_lo=1.0, p_hi=99.0):
    """Percentile stretch for display; preserves raw for saving."""
    lo = np.percentile(img16, p_lo); hi = np.percentile(img16, p_hi)
    if hi <= lo:  # degenerate; fallback to full range
        lo, hi = int(img16.min()), int(img16.max()) or 1
    x = np.clip((img16.astype(np.float32)-lo) / (hi-lo), 0, 1)
    return (x*255.0).astype(np.uint8)

def get_gray_and_raw(frame):
    """
    Returns (gray8_for_display, raw_for_save)
    raw_for_save is uint16 if we detected 16-bit input; otherwise uint8.
    """
    if frame is None: return None, None

    # Case A: single-channel
    if frame.ndim == 2:
        if frame.dtype == np.uint16:
            return stretch_u16_to_u8(frame), frame  # display, save raw16
        else:  # uint8 grayscale
            return frame, frame

    # Case B: 3-channel (likely BGR uint8)
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray8, gray8

    # Case C: unusual formats (very rare)
    if frame.ndim == 3 and frame.shape[2] == 1:
        ch = frame[:, :, 0]
        if ch.dtype == np.uint16:
            return stretch_u16_to_u8(ch), ch
        return ch, ch

    # Fallback
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return g, g

# Print what the driver actually granted (once)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera mode: {actual_w}x{actual_h}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed"); break

    gray8, raw = get_gray_and_raw(frame)
    if gray8 is None:
        print("Unsupported frame format"); break

    # x2 display (nearest neighbor = no smoothing)
    up = cv2.resize(gray8, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Camera Feed", up)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == 32:
        path = new_path()
        # Save raw16 if available; else save gray8
        if raw.dtype == np.uint16:
            cv2.imwrite(path, raw, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(path, gray8, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("Saved:", path)

cap.release(); cv2.destroyAllWindows()