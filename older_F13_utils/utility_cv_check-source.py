# Quickly check a platform capacity to capture a high-resolution stream from a source
# Break a source stream into portions, displayed individually and recombined. No interference.
 
import cv2
import numpy as np
import tkinter as tk

DISPLAY_SCALE = 0.5  # Scale factor for display (e.g., 0.5 for 50%, 1.25 for 125%)
SQUARE_PADDING = 20  # Padding in pixels between displayed squares
SQUARE_SIZE = 640  # Size of each square to be extracted from the frame
VIDEO_SOURCE = 0  # Can be "url" or "filepath" or "cuda=" device
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()  # Hide the root window (original stream in one piece)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def get_squares_from_frame(frame, square_size=SQUARE_SIZE):
    height, width, _ = frame.shape
    squares = []
    x_steps = width // square_size
    y_steps = height // square_size
    for y in range(y_steps):
        for x in range(x_steps):
            x_start = x * square_size
            y_start = y * square_size
            square = frame[y_start:y_start+square_size, x_start:x_start+square_size]
            squares.append(square)
    if height % square_size != 0:
        for x in range(x_steps):
            x_start = x * square_size
            y_start = height - square_size
            square = frame[y_start:y_start+square_size, x_start:x_start+square_size]
            squares.append(square)
    if width % square_size != 0:
        for y in range(y_steps):
            x_start = width - square_size
            y_start = y * square_size
            square = frame[y_start:y_start+square_size, x_start:x_start+square_size]
            squares.append(square)
    return squares

def draw_circle_on_square(square):
    height, width, _ = square.shape
    center = (width // 2, height // 2)
    cv2.circle(square, center, 50, (0, 0, 255), 5)
    return square

def combine_squares_to_frame(squares, frame_shape, square_size=SQUARE_SIZE):
    height, width, _ = frame_shape
    combined_frame = np.zeros((height, width, 3), dtype=np.uint8)
    x_steps = width // square_size
    y_steps = height // square_size
    index = 0
    for y in range(y_steps):
        for x in range(x_steps):
            if index < len(squares):
                x_start = x * square_size
                y_start = y * square_size
                combined_frame[y_start:y_start+square_size, x_start:x_start+square_size] = squares[index]
                index += 1
    if height % square_size != 0:
        for x in range(x_steps):
            if index < len(squares):
                x_start = x * square_size
                y_start = height - square_size
                combined_frame[y_start:y_start+square_size, x_start:x_start+square_size] = squares[index]
                index += 1
    if width % square_size != 0:
        for y in range(y_steps):
            if index < len(squares):
                x_start = width - square_size
                y_start = y * square_size
                combined_frame[y_start:y_start+square_size, x_start:x_start+square_size] = squares[index]
                index += 1
    return combined_frame

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    desired_width = STREAM_WIDTH
    desired_height = STREAM_HEIGHT
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Get the screen resolution
    screen_width, screen_height = get_screen_resolution()
    print(f"Detected screen resolution: {screen_width}x{screen_height}")

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Check the actual resolution set
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source stream resolution is set to: {actual_width}x{actual_height}")

    # Flag to control initial placement of windows
    initial_placement_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize the original frame to DISPLAY_SCALE factor
        original_resized = cv2.resize(frame, (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)))
        original_height, original_width = original_resized.shape[:2]

        # Calculate the center position for the original frame
        original_x = (screen_width - original_width) // 2
        original_y = (screen_height - original_height) // 2

        # Move and display the original frame in the center
        cv2.imshow('Original Feed', original_resized)
        if not initial_placement_done:
            cv2.moveWindow('Original Feed', original_x, original_y)

        # Extract squares from the original frame at full resolution without overlap
        squares = get_squares_from_frame(frame, square_size=SQUARE_SIZE)

        # Draw a red circle in the middle of each square
        squares_with_circles = [draw_circle_on_square(square.copy()) for square in squares]

        # Combine the squares back into a single frame
        combined_frame = combine_squares_to_frame(squares_with_circles, frame.shape, square_size=SQUARE_SIZE)

        # Resize and display the combined frame
        combined_resized = cv2.resize(combined_frame, (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)))
        cv2.imshow('Combined Feed', combined_resized)

        # Display the individual squares around the original frame
        num_squares_per_row = 3
        num_squares_per_col = 2

        # Calculate total width and height for all squares with padding
        total_width = num_squares_per_row * (int(SQUARE_SIZE * DISPLAY_SCALE) + SQUARE_PADDING) - SQUARE_PADDING
        total_height = num_squares_per_col * (int(SQUARE_SIZE * DISPLAY_SCALE) + SQUARE_PADDING) - SQUARE_PADDING

        # Calculate top-left corner to start displaying squares
        start_x = original_x + (original_width - total_width) // 2
        start_y = original_y + (original_height - total_height) // 2

        for i, square in enumerate(squares_with_circles):
            # Resize square for display using DISPLAY_SCALE
            resized_square = cv2.resize(square, (int(square.shape[1] * DISPLAY_SCALE), int(square.shape[0] * DISPLAY_SCALE)))

            # Calculate the position of the square relative to the starting point with padding
            x_offset = (i % num_squares_per_row) * (resized_square.shape[1] + SQUARE_PADDING)  # Horizontal position with padding
            y_offset = (i // num_squares_per_row) * (resized_square.shape[0] + SQUARE_PADDING)  # Vertical position with padding

            # Calculate the position on the screen
            square_x = start_x + x_offset
            square_y = start_y + y_offset

            # Create a named window and move it to the calculated position
            window_name = f'Square {i+1}'
            cv2.imshow(window_name, resized_square)
            if not initial_placement_done:
                cv2.moveWindow(window_name, square_x, square_y)

            # Stop after displaying the needed 6 squares
            if i == num_squares_per_row * num_squares_per_col - 1:
                break

        # Set the flag to True after the initial placement
        initial_placement_done = True

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
