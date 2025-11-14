# Review images in an input folder and extract regions of the fixed size to an output folder.
# Useful if images intended for training a model are in high resolution, while model is optimized 
# for lower-sized regions (640 x 640 is Ultralytics YOLO default).

import os
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import sys
import math
import time

INPUT_FOLDER = r"C:\\Users\\V\\Desktop\\In"
OUTPUT_FOLDER = r"C:\\Users\\V\\Desktop\\Out"

SQUARE_SIZE = 640  # Target CV model constraint 
RESAMPLING_ALG = Image.Resampling.BICUBIC  # Used for display only, regions saved without resizing
DISP_ASPECT_RATIO, LINE_COLOR, SAVED_COLOR = 0.8, "#f8931d", "#00ff00" # Gui settings
KEY_MAP = ['1234567890', 'qwertyuiop', 'asdfghjkl;', 'zxcvbnm,./']
FLAT_KEY_MAP = ''.join(KEY_MAP)
total_squares_saved = 0
processed_images_info = []
start_time = None

def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def filter_image_files(folder):
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        sys.exit(1)
    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_formats)], key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    if not files:
        print(f"Error: No valid image files found in '{folder}'.")
        sys.exit(1)
    return files

def ensure_output_folder_exists(folder):
    if not os.path.exists(folder):
        try: os.makedirs(folder)
        except OSError as e: sys.exit(1)

def draw_square(draw, top_left_x, top_left_y, bottom_right_x, bottom_right_y, key, color):
    center_x = (top_left_x + bottom_right_x) // 2
    center_y = (top_left_y + bottom_right_y) // 2
    square_radius = (bottom_right_x - top_left_x) // 2
    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline=color, width=1)
    draw.ellipse([(center_x - square_radius, center_y - square_radius), (center_x + square_radius, center_y + square_radius)], outline=color, width=2)
    draw.ellipse([(center_x - 30, center_y - 30), (center_x + 30, center_y + 30)], outline=color, width=2)
    draw.text((center_x, center_y), key, font=ImageFont.truetype("arial.ttf", size=32), fill=color, anchor='mm')

def calculate_rows_and_columns(img_width, img_height, square_size):
    num_columns = math.ceil(img_width / square_size)
    num_rows = math.ceil(img_height / square_size)
    if num_columns > 1: overlap_h = (square_size * num_columns - img_width) / (num_columns - 1)
    else: overlap_h = 0
    if num_rows > 1: overlap_v = (square_size * num_rows - img_height) / (num_rows - 1)
    else: overlap_v = 0
    return num_columns, num_rows, overlap_h, overlap_v

def calculate_squares(img_width, img_height, square_size, overlap_h, overlap_v):
    squares = []
    num_columns, num_rows, _, _ = calculate_rows_and_columns(img_width, img_height, square_size)
    for row in range(num_rows):
        for col in range(num_columns):
            top_left_x = col * (square_size - overlap_h)
            top_left_y = row * (square_size - overlap_v)
            bottom_right_x = min(top_left_x + square_size, img_width)
            bottom_right_y = min(top_left_y + square_size, img_height)
            if bottom_right_x <= top_left_x or bottom_right_y <= top_left_y:
                continue
            row_key_idx = row % len(KEY_MAP)
            key_row = KEY_MAP[row_key_idx]
            key = key_row[col % len(key_row)]
            squares.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y, key, row + 1, col + 1))
    return squares

def draw_squares_on_image(draw, squares, scale_x, scale_y, color):
    for square in squares:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y, key = square[:5]
        scaled_top_left_x = int(top_left_x * scale_x)
        scaled_top_left_y = int(top_left_y * scale_y)
        scaled_bottom_right_x = int(bottom_right_x * scale_x)
        scaled_bottom_right_y = int(bottom_right_y * scale_y)
        if scaled_bottom_right_x > scaled_top_left_x and scaled_bottom_right_y > scaled_top_left_y:
            draw_square(draw, scaled_top_left_x, scaled_top_left_y, scaled_bottom_right_x, scaled_bottom_right_y, key, color)

def draw_markup_on_image(img, original_width, original_height, display_width, display_height):
    resized_img = img.resize((display_width, display_height), RESAMPLING_ALG)
    draw = ImageDraw.Draw(resized_img)
    scale_x = display_width / original_width
    scale_y = display_height / original_height
    num_columns, num_rows, overlap_h, overlap_v = calculate_rows_and_columns(original_width, original_height, SQUARE_SIZE)
    squares = calculate_squares(original_width, original_height, SQUARE_SIZE, overlap_h, overlap_v)
    draw_squares_on_image(draw, squares, scale_x, scale_y, LINE_COLOR)
    img_tk = ImageTk.PhotoImage(resized_img)
    return squares, img_tk, draw, resized_img

def save_selected_squares(image, square_positions, selected_key, img_file, img_index):
    global total_squares_saved
    squares_saved = 0
    for square in square_positions:
        if square[4] == selected_key:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = square[:4]
            cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
            row_num, col_num = square[5], square[6]
            cropped_image_filename = f"i{img_index:04d}r{row_num:02d}c{col_num:02d}.png"
            cropped_image_path = os.path.join(OUTPUT_FOLDER, cropped_image_filename)
            cropped_image.save(cropped_image_path)
            squares_saved += 1
            total_squares_saved += 1
            print(f"Saved square at row {row_num}, column {col_num} -> {cropped_image_filename}")
    return squares_saved

def process_images():
    global start_time
    start_time = time.time()
    os.system('cls' if os.name == 'nt' else 'clear')
    screen_width, screen_height = get_screen_resolution()
    max_width = int(screen_width * DISP_ASPECT_RATIO)
    max_height = int(screen_height * DISP_ASPECT_RATIO)
    ensure_output_folder_exists(OUTPUT_FOLDER)
    image_files = filter_image_files(INPUT_FOLDER)
    total_images = len(image_files)
    current_image_index = 0
    root = tk.Tk()

    def display_next_image():
        nonlocal current_image_index
        if current_image_index >= total_images:
            root.quit()
            return
        img_file = image_files[current_image_index]
        img_path = os.path.join(INPUT_FOLDER, img_file)
        with Image.open(img_path) as img:
            original_width, original_height = img.size
            aspect_ratio = min(max_width / original_width, max_height / original_height)
            display_width = int(original_width * aspect_ratio)
            display_height = int(original_height * aspect_ratio)
            print(f"\nProcessing image {current_image_index + 1}/{total_images}: {img_file} - Resolution: {original_width}x{original_height}")
            root.geometry(f"{display_width}x{display_height}")
            root.focus_force()
            for widget in root.winfo_children():
                widget.destroy()
            square_positions, img_tk, draw, resized_img = draw_markup_on_image(img, original_width, original_height, display_width, display_height)
            label = tk.Label(root, image=img_tk)
            label.img_tk = img_tk
            label.pack()

            def on_key_press(event):
                nonlocal current_image_index
                key = event.char
                squares_saved_accumulated = 0
                if key == ' ':
                    processed_images_info.append((img_file, f"{original_width}x{original_height}", squares_saved_accumulated))  
                    current_image_index += 1
                    display_next_image()
                elif key == '\x1b':
                    root.quit()
                elif key in FLAT_KEY_MAP:
                    squares_saved = save_selected_squares(img, square_positions, key, img_file, current_image_index)
                    squares_saved_accumulated += squares_saved
                    draw_squares_on_image(draw, [sq for sq in square_positions if sq[4] == key], display_width / original_width, display_height / original_height, SAVED_COLOR)
                    img_tk = ImageTk.PhotoImage(resized_img)
                    label.config(image=img_tk)
                    label.img_tk = img_tk
                if squares_saved_accumulated > 0 and current_image_index < len(image_files) - 1:
                    processed_images_info.append((img_file, f"{original_width}x{original_height}", squares_saved_accumulated))
            root.bind("<KeyPress>", on_key_press)

    display_next_image()
    root.mainloop()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    avg_time_per_square = total_time / total_squares_saved if total_squares_saved > 0 else 0
    print(f"\nProcessed folder: {INPUT_FOLDER}")
    print(f"\nTotal valid images: {total_images}")
    print("\nFinal Report:")
    for info in processed_images_info:
        file_name, resolution, squares_saved = info
        if squares_saved == 0: print(f"File: {file_name} - {resolution} - skipped")
        else: print(f"File: {file_name} - {resolution} - {squares_saved} squares saved")
    print(f"\nTotal squares saved: {total_squares_saved}")
    print(f"Process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Process ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Average time per saved square: {avg_time_per_square:.2f} seconds")

if __name__ == "__main__": process_images()
