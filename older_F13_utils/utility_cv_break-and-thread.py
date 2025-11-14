# Quickly check a platform capacity to run interference on multiple regions of a high-resolution stream
# Break a source stream into portions > instantiate model for each "square" > interfere > recombine and plot

import cv2
import numpy as np
import threading
import tkinter as tk
from ultralytics import YOLO

VIDEO_SOURCE = 0
CAP_WIDTH = 1920
CAP_HEIGHT = 1080
DISPLAY_SCALE = 0.5
SQUARE_PADDING = 20
SQUARE_SIZE = 640
CONFIDENCE = 0.4

square_feeds = []  # List to hold individual video feed frames
processed_feeds = []  # List to hold YOLO processed feeds
stop_event = threading.Event()  # Event to stop threads

def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def get_squares_info(frame_shape, square_size=SQUARE_SIZE):
    height, width, _ = frame_shape
    x_steps = (width + square_size - 1) // square_size
    y_steps = (height + square_size - 1) // square_size
    return x_steps, y_steps

def update_square_feeds(cap, x_steps, y_steps):
    global square_feeds
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        square_feeds = []
        for y in range(y_steps):
            for x in range(x_steps):
                x_start = x * SQUARE_SIZE
                y_start = y * SQUARE_SIZE
                square_frame = frame[y_start:y_start+SQUARE_SIZE, x_start:x_start+SQUARE_SIZE].copy()
                square_feeds.append(square_frame)

def process_square_with_yolo(model, index):
    global processed_feeds
    while not stop_event.is_set():
        if square_feeds and index < len(square_feeds):
            square = square_feeds[index].copy()
            results = model(square, conf=CONFIDENCE)
            processed_square = results[0].plot()
            height, width, _ = processed_square.shape
            center = (width // 2, height // 2)
            cv2.circle(processed_square, center, 50, (0, 0, 255), 5)
            processed_feeds[index] = processed_square

def combine_feeds_to_frame(feeds, frame_shape, x_steps, y_steps, square_size=SQUARE_SIZE):
    height, width, _ = frame_shape
    combined_frame = np.zeros((height, width, 3), dtype=np.uint8)
    index = 0
    for y in range(y_steps):
        for x in range(x_steps):
            if index < len(feeds) and feeds[index] is not None:
                x_start = x * square_size
                y_start = y * square_size
                combined_frame[y_start:y_start+square_size, x_start:x_start+square_size] = feeds[index]
                index += 1
    return combined_frame

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    desired_width = CAP_WIDTH 
    desired_height = CAP_HEIGHT 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    screen_width, screen_height = get_screen_resolution()
    print(f"Detected screen resolution: {screen_width}x{screen_height}")
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    x_steps, y_steps = get_squares_info(frame.shape, square_size=SQUARE_SIZE)  # Calculate the number of squares based on sizes
    total_squares = x_steps * y_steps
    global processed_feeds
    processed_feeds = [None] * total_squares  # Initialize processed feeds list
    yolo_models = [YOLO('yolo11n.pt') for _ in range(total_squares)]  # Create YOLO models for each square
    feed_thread = threading.Thread(target=update_square_feeds, args=(cap, x_steps, y_steps))
    feed_thread.start()  # Start a thread to update square feeds continuously
    threads = []  # Start a thread for each square to process with YOLO
    for i in range(total_squares):
        thread = threading.Thread(target=process_square_with_yolo, args=(yolo_models[i], i))
        threads.append(thread)
        thread.start()
    while True: # Display the original feed
        ret, frame = cap.read()  # Update the original feed in each loop
        if not ret: break
        original_resized = cv2.resize(frame, (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)))
        cv2.imshow('Original Feed', original_resized)
        # Combine the square feeds into a single frame, resize and display the combined
        combined_frame = combine_feeds_to_frame(processed_feeds, frame.shape, x_steps, y_steps, square_size=SQUARE_SIZE)
        combined_resized = cv2.resize(combined_frame, (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)))
        cv2.imshow('Combined Feed', combined_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    # Stop all threads and release resources
    stop_event.set()
    feed_thread.join()
    for thread in threads: thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
