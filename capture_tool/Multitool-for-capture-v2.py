import cv2
import os
import shutil
import numpy as np
import random
import string
import time

# ================== ПУТИ ==================
SRC = r"C:\Users\User\Downloads\captures"
DST = r"C:\Users\User\Downloads\16.11.sorted"

os.makedirs(DST, exist_ok=True)

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

files = [f for f in os.listdir(SRC) if os.path.splitext(f)[1].lower() in EXT]
files.sort()

if not files:
    raise SystemExit("Нет изображений в папке: " + SRC)

index = 0
picked = 0
total = len(files)

# ================== UI РАЗМЕРЫ ==================
PANEL_W = 320            # правая панель
THUMB_W = 80             # ширина миниатюры
THUMB_H = 80             # высота миниатюры
THUMB_BAR_H = THUMB_H + 20

# размеры области просмотра картинки (зависят от изображения)
main_w = 0
main_h = 0

# ================== СОСТОЯНИЕ ПРОСМОТРА ==================
interaction_mode = "VIEW"   # VIEW / CROP
view_scale = 1.0            # текущий масштаб (анимируемый)
target_view_scale = 1.0     # целевой масштаб для плавного зума
view_cx = 0.0               # текущий центр окна просмотра (в координатах оригинала)
view_cy = 0.0
target_view_cx = 0.0        # целевой центр (для плавного зума)
target_view_cy = 0.0
view_x0 = 0.0               # видимый диапазон в оригинале
view_y0 = 0.0
view_x1 = 0.0
view_y1 = 0.0

ZOOM_ANIM_SPEED = 0.25      # скорость анимации зума (0..1)

dragging = False            # перетаскивание RMB при зуме
drag_start = (0, 0)

show_grid = False           # показывать сетку "правило третей"
show_original_view = False  # временный показ оригинала (TAB)
loop_navigation = True      # переход с конца в начало и наоборот

# ================== CROP ==================
crop_start = None           # (x, y) в координатах display (0..main_w-1, 0..main_h-1)
crop_end = None
cropping = False

# ================== КЭШИ ==================
rotated_cache = {}          # fname -> img
cropped_cache = {}          # fname -> img
thumb_cache = {}            # index -> thumb

image_cache = {}            # fname -> исходный img
image_cache_order = []      # порядок ключей для простого LRU
IMAGE_CACHE_MAX = 32

window_centered = False     # чтобы центрировать окно один раз

# ================== ДРУГОЕ СОСТОЯНИЕ ==================
hover_thumb_index = None    # индекс миниатюры под мышкой
last_lclick_time = 0        # для простого детекта даблклика ЛКМ
DOUBLE_CLICK_INTERVAL = 0.35  # секунд


# ================== УТИЛИТЫ ==================
def random_prefix(n=8):
    return ''.join(random.choices(string.ascii_uppercase, k=n))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cache_put_image(fname, img):
    """Положить изображение в кеш с простым LRU."""
    global image_cache, image_cache_order
    image_cache[fname] = img
    if fname in image_cache_order:
        image_cache_order.remove(fname)
    image_cache_order.append(fname)

    if len(image_cache_order) > IMAGE_CACHE_MAX:
        old = image_cache_order.pop(0)
        image_cache.pop(old, None)


def get_base_image(fname):
    """Базовое изображение: из кеша или с диска."""
    if fname in image_cache:
        return image_cache[fname]

    path = os.path.join(SRC, fname)
    img = cv2.imread(path)
    if img is not None:
        cache_put_image(fname, img)
    return img


def get_original_img():
    """Текущее изображение: кропнутое, повернутое или оригинал."""
    fname = files[index]

    if fname in cropped_cache:
        return cropped_cache[fname]
    if fname in rotated_cache:
        return rotated_cache[fname]

    return get_base_image(fname)


def init_view_for_image(img):
    """Сброс зума/окна под новое изображение."""
    global main_w, main_h
    global view_scale, target_view_scale
    global view_cx, view_cy, target_view_cx, target_view_cy
    global view_x0, view_y0, view_x1, view_y1

    h, w = img.shape[:2]
    main_w = w
    main_h = h

    view_scale = 1.0
    target_view_scale = 1.0

    view_cx = w / 2.0
    view_cy = h / 2.0
    target_view_cx = view_cx
    target_view_cy = view_cy

    view_x0, view_y0 = 0.0, 0.0
    view_x1, view_y1 = float(w), float(h)
    # если меняется размер основной области, позволим окну пересчитать размер
    global window_centered
    window_centered = False


def step_zoom_animation():
    """Плавное приближение текущего масштаба/центра к целевым значениям."""
    global view_scale, view_cx, view_cy

    if abs(target_view_scale - view_scale) > 1e-3:
        view_scale += (target_view_scale - view_scale) * ZOOM_ANIM_SPEED
    else:
        view_scale = target_view_scale

    if abs(target_view_cx - view_cx) > 0.1:
        view_cx += (target_view_cx - view_cx) * ZOOM_ANIM_SPEED
    else:
        view_cx = target_view_cx

    if abs(target_view_cy - view_cy) > 0.1:
        view_cy += (target_view_cy - view_cy) * ZOOM_ANIM_SPEED
    else:
        view_cy = target_view_cy


def update_view_window(img):
    """
    На основе view_scale / view_cx / view_cy вычисляем участок
    оригинального изображения, который надо показать, и растягиваем
    его под main_w x main_h.
    """
    global view_x0, view_y0, view_x1, view_y1

    h, w = img.shape[:2]

    # защита от дурацких размеров
    if main_w <= 0 or main_h <= 0:
        return img

    # окно просмотра в координатах оригинала
    win_w = main_w / max(view_scale, 1e-3)
    win_h = main_h / max(view_scale, 1e-3)

    cx = clamp(view_cx, win_w / 2, max(w - win_w / 2, win_w / 2))
    cy = clamp(view_cy, win_h / 2, max(h - win_h / 2, win_h / 2))

    x0 = cx - win_w / 2
    y0 = cy - win_h / 2
    x1 = x0 + win_w
    y1 = y0 + win_h

    x0 = clamp(x0, 0, max(w - 1, 1))
    y0 = clamp(y0, 0, max(h - 1, 1))
    x1 = clamp(x1, x0 + 1, w)
    y1 = clamp(y1, y0 + 1, h)

    view_x0, view_y0, view_x1, view_y1 = x0, y0, x1, y1

    crop = img[int(y0):int(y1), int(x0):int(x1)]
    display_img = cv2.resize(crop, (main_w, main_h), interpolation=cv2.INTER_LINEAR)
    return display_img


def zoom_at(display_x, display_y, factor):
    """Задать новый целевой зум вокруг точки под курсором (display_x, display_y)."""
    global target_view_scale, target_view_cx, target_view_cy

    img = get_original_img()
    if img is None or main_w <= 0 or main_h <= 0:
        return

    new_scale = clamp(target_view_scale * factor, 1.0, 8.0)
    if abs(new_scale - target_view_scale) < 1e-3:
        return

    # текущий участок
    x0, y0, x1, y1 = view_x0, view_y0, view_x1, view_y1

    # точка под курсором в координатах оригинала
    tx = x0 + (display_x / float(main_w)) * (x1 - x0)
    ty = y0 + (display_y / float(main_h)) * (y1 - y0)

    target_view_scale = new_scale
    target_view_cx = tx
    target_view_cy = ty


# ================== THUMBNAILS ==================
def load_thumb(i):
    if i in thumb_cache:
        return thumb_cache[i]

    fname = files[i]
    img = get_base_image(fname)
    if img is None:
        thumb_cache[i] = None
        return None

    h, w = img.shape[:2]
    scale = THUMB_H / float(h)
    tw = max(int(w * scale), 1)

    resized = cv2.resize(img, (tw, THUMB_H), interpolation=cv2.INTER_AREA)
    thumb = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    thumb[:] = (25, 25, 25)

    x_off = max((THUMB_W - tw) // 2, 0)
    thumb[:, x_off:x_off + min(tw, THUMB_W)] = resized[:, :min(tw, THUMB_W)]

    thumb_cache[i] = thumb
    return thumb


def thumb_visible_range():
    slot = THUMB_W + 10
    if main_w <= 0:
        return 0, 1, slot

    max_vis = max(1, (main_w - 20) // slot)
    start = max(0, index - max_vis // 2)
    if start + max_vis > total:
        start = max(0, total - max_vis)

    return start, max_vis, slot


def draw_thumbs(canvas, base_y):
    start, count, slot = thumb_visible_range()
    x = 10

    for i in range(start, start + count):
        if i >= total:
            break
        t = load_thumb(i)
        if t is not None:
            canvas[base_y:base_y + THUMB_H, x:x + THUMB_W] = t

        # рамка активной миниатюры
        if i == index:
            cv2.rectangle(canvas, (x - 2, base_y - 2),
                          (x + THUMB_W + 2, base_y + THUMB_H + 2),
                          (0, 255, 255), 2)
        # рамка при наведении
        elif i == hover_thumb_index:
            cv2.rectangle(canvas, (x - 1, base_y - 1),
                          (x + THUMB_W + 1, base_y + THUMB_H + 1),
                          (0, 180, 255), 1)

        x += slot


# ================== ПАНЕЛЬ СПРАВА ==================
def draw_panel(canvas):
    h, w = canvas.shape[:2]
    x0 = main_w

    # фон панели
    cv2.rectangle(canvas, (x0, 0), (w, h), (30, 30, 30), -1)

    # subtle translucent backdrop for improved readability
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 + 8, 30), (w - 8, h - 8), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

    # цвета (тёплый/умеренно неоновый акцент и более контрастный текст)
    neon = (80, 200, 100)
    soft = (120, 200, 140)
    key_bg = (60, 60, 60)
    key_border = (120, 200, 140)
    text_col = (220, 220, 220)

    # ===== Заголовок =====
    cv2.putText(canvas, "MULTITOOL", (x0 + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, neon, 2)

    # ===== Режим =====
    if interaction_mode == "CROP":
        mode_txt = "CROP"
        col = (0, 220, 255)
    elif view_scale > 1.01:
        mode_txt = "ZOOM"
        col = (255, 200, 0)
    else:
        mode_txt = "VIEW"
        col = (0, 200, 0)

    cv2.putText(canvas, f"MODE: {mode_txt}", (x0 + 20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

    # ===== INFO =====
    cv2.putText(canvas, f"{index + 1}/{total}", (x0 + 20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, soft, 2)
    cv2.putText(canvas, f"Picked: {picked}", (x0 + 20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, soft, 2)
    cv2.putText(canvas, f"Zoom: {view_scale:.2f}x", (x0 + 20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, soft, 2)

    grid_txt = "ON" if show_grid else "OFF"
    cv2.putText(canvas, f"Grid: {grid_txt}", (x0 + 20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, soft, 2)

    # ===== Разделитель =====
    cv2.line(canvas, (x0 + 10, 230), (w - 10, 230), (60, 60, 60), 1)

    # =====================================================================
    #                          НОВЫЕ HOTKEYS
    # =====================================================================

    # список хоткеев в формате:
    # ("KEY", "description")
    hotkeys = [
        ("SPACE", "Select & Save"),
        ("A / D", "Prev / Next"),
        ("W / S", "+10 / -10"),
        ("PgUp / PgDn", "+50 / -50"),
        ("R", "Rotate 90 CCW"),
        ("C", "Crop mode"),
        ("ENTER", "Apply crop"),
        ("Z", "Reset zoom"),
        ("G", "Toggle grid"),
        ("TAB", "Original view"),
        ("Q", "Reset view"),
        ("ESC", "Exit / Cancel crop"),
        ("Wheel", "Zoom"),
        ("RMB", "Pan (drag)"),
        ("Double LMB", "Zoom x2"),
    ]

    # параметры
    start_y = 255
    line_h = 32
    key_pad_x = 14
    key_pad_y = 6

    for key_name, desc in hotkeys:
        # рисуем капсулу для кнопки
        x_key = x0 + 20
        y_key = start_y

        # размеры текста
        (tw, th), _ = cv2.getTextSize(key_name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        box_w = tw + key_pad_x * 2
        box_h = th + key_pad_y * 2

        # фон кнопки
        cv2.rectangle(canvas,
                      (x_key, y_key),
                      (x_key + box_w, y_key + box_h),
                      key_bg, -1)

        # граница кнопки
        cv2.rectangle(canvas,
                      (x_key, y_key),
                      (x_key + box_w, y_key + box_h),
                      key_border, 1)

        # текст кнопки
        cv2.putText(canvas, key_name,
                    (x_key + key_pad_x, y_key + box_h - key_pad_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 1)

        # описание справа от кнопки
        cv2.putText(canvas, desc,
                    (x_key + box_w + 12, y_key + box_h - key_pad_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, neon, 1)

        start_y += line_h

# ================== OVERLAYS ==================
def draw_rule_of_thirds(display_img):
    """Рисуем сетку 'правило третей'."""
    if main_w <= 0 or main_h <= 0:
        return
    color = (80, 220, 80)
    thickness = 1
    # вертикали
    x1 = main_w // 3
    x2 = 2 * main_w // 3
    cv2.line(display_img, (x1, 0), (x1, main_h), color, thickness)
    cv2.line(display_img, (x2, 0), (x2, main_h), color, thickness)
    # горизонтали
    y1 = main_h // 3
    y2 = 2 * main_h // 3
    cv2.line(display_img, (0, y1), (main_w, y1), color, thickness)
    cv2.line(display_img, (0, y2), (main_w, y2), color, thickness)


def draw_crop_on_display(display_img):
    """Рисуем рамку кропа и затемнение вне области."""
    if interaction_mode != "CROP" or not crop_start or not crop_end:
        return

    x1, y1 = crop_start
    x2, y2 = crop_end
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x1 = clamp(x1, 0, max(main_w - 1, 0))
    x2 = clamp(x2, 0, max(main_w - 1, 0))
    y1 = clamp(y1, 0, max(main_h - 1, 0))
    y2 = clamp(y2, 0, max(main_h - 1, 0))

    if x2 <= x1 or y2 <= y1:
        return

    # затемнение вне области
    overlay = display_img.copy()
    overlay[:] = (0, 0, 0)
    overlay[y1:y2, x1:x2] = display_img[y1:y2, x1:x2]
    cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)

    # яркая рамка
    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # ручки по краям
    handle_size = 6
    points = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
        ((x1 + x2) // 2, y1),
        ((x1 + x2) // 2, y2),
        (x1, (y1 + y2) // 2),
        (x2, (y1 + y2) // 2),
    ]
    for (hx, hy) in points:
        cv2.rectangle(display_img,
                      (hx - handle_size, hy - handle_size),
                      (hx + handle_size, hy + handle_size),
                      (255, 255, 255), -1)


def draw_main_frame(canvas):
    """Белая/цветная рамка вокруг основной области просмотра."""
    if main_w <= 0 or main_h <= 0:
        return

    if interaction_mode == "CROP":
        color = (0, 220, 255)   # голубой
    elif view_scale > 1.01:
        color = (255, 220, 0)   # желтый
    else:
        color = (240, 240, 240)  # белый

    # внешняя рамка
    cv2.rectangle(canvas, (0, 0), (main_w - 1, main_h - 1), color, 4)
    # легкая "тень"
    cv2.rectangle(canvas, (3, 3), (main_w - 4, main_h - 4), (50, 50, 50), 1)


# ================== ОТРИСОВКА ==================
def show():
    global window_centered, view_x0, view_y0, view_x1, view_y1

    img = get_original_img()
    if img is None:
        return

    global main_w, main_h
    if main_w == 0 or main_h == 0:
        init_view_for_image(img)

    # плавный зум
    step_zoom_animation()

    # TAB: показать оригинал (игнорируя зум)
    if show_original_view:
        display = cv2.resize(img, (main_w, main_h), interpolation=cv2.INTER_LINEAR)
        # чтобы кроп после TAB имел полный диапазон
        h, w = img.shape[:2]
        view_x0, view_y0 = 0.0, 0.0
        view_x1, view_y1 = float(w), float(h)
    else:
        display = update_view_window(img)

    # копия для оверлеев
    disp_with_overlay = display.copy()

    # кроп
    draw_crop_on_display(disp_with_overlay)

    # сетка
    if show_grid and interaction_mode != "CROP":
        draw_rule_of_thirds(disp_with_overlay)

    canvas_h = main_h + THUMB_BAR_H
    canvas_w = main_w + PANEL_W

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)

    canvas[0:main_h, 0:main_w] = disp_with_overlay

    # рамка вокруг основного окна
    draw_main_frame(canvas)

    draw_panel(canvas)
    draw_thumbs(canvas, main_h + 5)

    cv2.imshow("MULTITOOL", canvas)

    # центрируем окно один раз (под монитор 1920x1080)
    if not window_centered:
        screen_w, screen_h = 1920, 1080
        pos_x = max((screen_w - canvas_w) // 2, 0)
        pos_y = max((screen_h - canvas_h) // 2, 0)
        try:
            cv2.resizeWindow("MULTITOOL", canvas_w, canvas_h)
            cv2.moveWindow("MULTITOOL", pos_x, pos_y)
        except Exception:
            pass
        window_centered = True


def set_image_index(new_index):
    global index, interaction_mode, show_original_view
    new_index = int(clamp(new_index, 0, total - 1))
    index = new_index
    interaction_mode = "VIEW"
    show_original_view = False
    img = get_original_img()
    if img is not None:
        init_view_for_image(img)


# ================== MOUSE CALLBACK ==================
def mouse_cb(event, x, y, flags, param):
    global interaction_mode, crop_start, crop_end, cropping
    global dragging, drag_start, view_cx, view_cy
    global hover_thumb_index, last_lclick_time
    global target_view_cx, target_view_cy, target_view_scale

    img = get_original_img()
    if img is None or main_w == 0 or main_h == 0:
        return

    h, w = img.shape[:2]

    # зона thumbnails
    if y >= main_h:
        hover_thumb_index = None
        if x < main_w:
            start, count, slot = thumb_visible_range()
            rel = x - 10
            if rel >= 0:
                slot_i = rel // slot
                if 0 <= slot_i < count:
                    hover_thumb_index = start + slot_i

            # клик по миниатюре
            if event == cv2.EVENT_LBUTTONDOWN and hover_thumb_index is not None:
                if hover_thumb_index < total:
                    set_image_index(hover_thumb_index)
        return

    # зона панели
    if x >= main_w:
        hover_thumb_index = None
        return

    # === CROP MODE ===
    if interaction_mode == "CROP":
        if event == cv2.EVENT_LBUTTONDOWN:
            crop_start = (x, y)
            crop_end = (x, y)
            cropping = True
        elif event == cv2.EVENT_MOUSEMOVE and cropping:
            crop_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and cropping:
            crop_end = (x, y)
            cropping = False
        return

    # === ZOOM (колесо) ===
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_at(x, y, 1.25)
        else:
            zoom_at(x, y, 1.0 / 1.25)
        return

    # === Double LMB для быстрой смены масштаба ===
    if event == cv2.EVENT_LBUTTONDOWN:
        now = time.time()
        if now - last_lclick_time < DOUBLE_CLICK_INTERVAL:
            # double click
            if target_view_scale <= 1.01:
                zoom_at(x, y, 2.0)
            else:
                # сброс к 1.0
                target_view_scale = 1.0
                # центрируем на текущей точке
                x0, y0, x1, y1 = view_x0, view_y0, view_x1, view_y1
                tx = x0 + (x / float(main_w)) * (x1 - x0)
                ty = y0 + (y / float(main_h)) * (y1 - y0)
                target_view_cx = tx
                target_view_cy = ty
        last_lclick_time = now

    # === PAN (правая кнопка) ===
    if view_scale > 1.01:
        if event == cv2.EVENT_RBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            shift_x = dx / max(view_scale, 1e-3)
            shift_y = dy / max(view_scale, 1e-3)
            view_cx = clamp(view_cx - shift_x, 0, w)
            view_cy = clamp(view_cy - shift_y, 0, h)
            target_view_cx = view_cx
            target_view_cy = view_cy
            drag_start = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            dragging = False


# ================== MAIN LOOP ==================
cv2.namedWindow("MULTITOOL", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("MULTITOOL", mouse_cb)

init_view_for_image(get_original_img())

running = True
while running:
    show()
    key = cv2.waitKey(20) & 0xFFFF   # 20 мс — чтобы мышь перерисовывала кроп

    if key == 0xFFFF or key == 65535:  # ничего не нажато
        continue

    # ---- ЗУМ КЛАВИШАМИ (+ / -) ----
    if key in (ord('+'), ord('=')):
        zoom_at(main_w // 2, main_h // 2, 1.25)

    elif key in (ord('-'), ord('_')):
        zoom_at(main_w // 2, main_h // 2, 1.0 / 1.25)

    # ---- SPACE: выбрать и сохранить ----
    elif key == 32:
        fname = files[index]
        pref = random_prefix(8)
        name, ext = os.path.splitext(fname)
        new_name = f"{pref}_{name}{ext}"

        # Сохраняем текущее изображение (включая кроп/поворот), если доступно.
        # Всегда масштабируем до 640x480.
        disp_img = get_original_img()
        out_path = os.path.join(DST, new_name)
        if disp_img is not None:
            try:
                # Масштабируем до 640x480
                resized_img = cv2.resize(disp_img, (640, 480), interpolation=cv2.INTER_AREA)
                cv2.imwrite(out_path, resized_img)
            except Exception:
                # fallback: копируем оригинал
                shutil.copy2(os.path.join(SRC, fname), out_path)
        else:
            shutil.copy2(os.path.join(SRC, fname), out_path)

        picked += 1
        # переход дальше
        if index < total - 1:
            set_image_index(index + 1)
        elif loop_navigation:
            set_image_index(0)

    # ---- ROTATE ----
    elif key in (ord('r'), ord('R')):
        img = get_original_img()
        if img is not None:
            rotated_cache[files[index]] = cv2.rotate(
                img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            init_view_for_image(rotated_cache[files[index]])

    # ---- CROP MODE ----
    elif key in (ord('c'), ord('C')):
        interaction_mode = "CROP"
        crop_start = None
        crop_end = None
        cropping = False

    # ---- APPLY CROP (ENTER) ----
    elif key == 13 and interaction_mode == "CROP":
        img = get_original_img()
        if img is not None and crop_start and crop_end:
            xs, ys = crop_start
            xe, ye = crop_end
            xs, xe = sorted([clamp(xs, 0, main_w - 1),
                             clamp(xe, 0, main_w - 1)])
            ys, ye = sorted([clamp(ys, 0, main_h - 1),
                             clamp(ye, 0, main_h - 1)])

            if (xe - xs) > 2 and (ye - ys) > 2:
                # пересчёт в координаты оригинала
                x0_orig = view_x0 + (xs / main_w) * (view_x1 - view_x0)
                x1_orig = view_x0 + (xe / main_w) * (view_x1 - view_x0)
                y0_orig = view_y0 + (ys / main_h) * (view_y1 - view_y0)
                y1_orig = view_y0 + (ye / main_h) * (view_y1 - view_y0)

                x0_i = int(clamp(x0_orig, 0, img.shape[1] - 1))
                x1_i = int(clamp(x1_orig, 0, img.shape[1] - 1))
                y0_i = int(clamp(y0_orig, 0, img.shape[0] - 1))
                y1_i = int(clamp(y1_orig, 0, img.shape[0] - 1))

                x0_i, x1_i = sorted([x0_i, x1_i])
                y0_i, y1_i = sorted([y0_i, y1_i])

                if x1_i > x0_i and y1_i > y0_i:
                    cropped = img[y0_i:y1_i, x0_i:x1_i]
                    cropped_cache[files[index]] = cropped
                    init_view_for_image(cropped)

        interaction_mode = "VIEW"
        crop_start = None
        crop_end = None
        cropping = False

    # ---- ESC ----
    elif key == 27:
        if interaction_mode == "CROP":
            interaction_mode = "VIEW"
            crop_start = None
            crop_end = None
            cropping = False
        else:
            running = False

    # ---- NEXT ----
    elif key in (ord('d'), ord('D'), 83):  # 83 = RIGHT ARROW при & 0xFF
        if index < total - 1:
            set_image_index(index + 1)
        elif loop_navigation:
            set_image_index(0)

    # ---- PREVIOUS ----
    elif key in (ord('a'), ord('A'), 81):  # 81 = LEFT ARROW при & 0xFF
        if index > 0:
            set_image_index(index - 1)
        elif loop_navigation:
            set_image_index(total - 1)

    # ---- Быстрый шаг: W/S или стрелки вверх/вниз ----
    # W или стрелка вверх → вперёд на 10
    elif key in (ord('w'), ord('W'), 82):  # 82 = UP ARROW
        set_image_index(index + 10)

    # S или стрелка вниз → назад на 10
    elif key in (ord('s'), ord('S'), 84):  # 84 = DOWN ARROW
        set_image_index(index - 10)

    # ---- PageUp/PageDown: +/-50 ----
    elif key == 2162688:  # PageUp (часто такое значение)
        set_image_index(index - 50  if index - 50 >= 0 else 0)

    elif key == 2228224:  # PageDown
        set_image_index(index + 50 if index + 50 < total else total - 1)

    # ---- RESET ZOOM ----
    elif key in (ord('z'), ord('Z')):
        img = get_original_img()
        if img is not None:
            init_view_for_image(img)

    # ---- TOGGLE GRID ----
    elif key in (ord('g'), ord('G')):
        show_grid = not show_grid

    # ---- TOGGLE ORIGINAL VIEW (TAB) ----
    elif key == 9:
        show_original_view = not show_original_view

    # ---- RESET VIEW (Q) ----
    elif key in (ord('q'), ord('Q')):
        img = get_original_img()
        if img is not None:
            init_view_for_image(img)
        interaction_mode = "VIEW"
        crop_start = None
        crop_end = None
        cropping = False
        show_grid = False
        show_original_view = False

cv2.destroyAllWindows()
