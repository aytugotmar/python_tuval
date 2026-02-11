"""
Air Gesture Drawing System - Simplified Stable Version
Mac Fullscreen Fix Included
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math

# ==========================================================
# CONFIG
# ==========================================================

TARGET_FPS = 60
BRUSH_THICKNESS = 4
ERASER_SIZE = 70

CURSOR_SENS = 1.1
HAND_LOST_TIMEOUT = 0.3
ROCK_COOLDOWN = 0.7  # renk değişim spam engeli

MIN_DETECTION_CONF = 0.8
MIN_TRACKING_CONF = 0.8

# ==========================================================
# INITIALIZATION
# ==========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
    max_num_hands=1
)

cap = cv2.VideoCapture(0)

success, frame_init = cap.read()
if not success:
    raise RuntimeError("Kamera açılamadı.")

height, width, _ = frame_init.shape

# 🔥 MAC TRUE FULLSCREEN FIX
cv2.namedWindow("AirCanvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AirCanvas", width, height)
cv2.setWindowProperty("AirCanvas",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

# ==========================================================
# STATE
# ==========================================================

previous_point = None
last_hand_seen_time = time.time()
last_rock_time = 0

color_palette = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255)
]

color_index = 0
draw_color = color_palette[color_index]

previous_time = time.time()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def calculate_distance(p1, p2):
    return int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

def is_finger_up(tip, dip):
    return tip[1] < dip[1]

def map_coordinates(lm_x, lm_y):
    center_x = width // 2
    center_y = height // 2

    x = int(center_x + (lm_x - 0.5) * width * CURSOR_SENS)
    y = int(center_y + (lm_y - 0.5) * height * CURSOR_SENS)

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    return x, y

# ==========================================================
# MAIN LOOP
# ==========================================================

while cap.isOpened():

    now = time.time()
    elapsed = now - previous_time
    if elapsed < 1.0 / TARGET_FPS:
        time.sleep((1.0 / TARGET_FPS) - elapsed)
    previous_time = time.time()

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    display_canvas = canvas.copy()

    # Kamera preview (küçük)
    preview_w = int(width * 0.22)
    preview_h = int(height * 0.22)
    small_frame = cv2.resize(frame, (preview_w, preview_h))
    display_canvas[0:preview_h,
                   width - preview_w:width] = small_frame

    if results.multi_hand_landmarks:

        last_hand_seen_time = time.time()

        hand_landmarks = results.multi_hand_landmarks[0]

        points = []
        for lm in hand_landmarks.landmark:
            x, y = map_coordinates(lm.x, lm.y)
            points.append((x, y))

        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]

        index_dip = points[6]
        middle_dip = points[10]
        ring_dip = points[14]
        pinky_dip = points[18]

        index_up = is_finger_up(index_tip, index_dip)
        middle_up = is_finger_up(middle_tip, middle_dip)
        ring_up = is_finger_up(ring_tip, ring_dip)
        pinky_up = is_finger_up(pinky_tip, pinky_dip)

        # ✋ TÜM EL AÇIK → SİLGİ
        if index_up and middle_up and ring_up and pinky_up:

            previous_point = None
            cv2.circle(canvas,
                       index_tip,
                       ERASER_SIZE,
                       (255, 255, 255),
                       cv2.FILLED)

        # 🤘 ROCK → RENK DEĞİŞTİR
        elif index_up and pinky_up and not middle_up and not ring_up:

            if time.time() - last_rock_time > ROCK_COOLDOWN:
                color_index = (color_index + 1) % len(color_palette)
                draw_color = color_palette[color_index]
                last_rock_time = time.time()

            previous_point = None

        # 🤏 PINCH → KALEM MODU
        else:
            pinch_distance = calculate_distance(thumb_tip, index_tip)

            pinch_active = pinch_distance < 35

            draw_mode = (index_up and not middle_up and not ring_up and not pinky_up) or pinch_active

            if draw_mode:

                if previous_point is None:
                    previous_point = index_tip
                else:
                    cv2.line(canvas,
                             previous_point,
                             index_tip,
                             draw_color,
                             BRUSH_THICKNESS)
                    previous_point = index_tip
            else:
                previous_point = None

        # İmleç
        cv2.circle(display_canvas, index_tip, 8, (0, 0, 0), cv2.FILLED)

    else:
        if time.time() - last_hand_seen_time > HAND_LOST_TIMEOUT:
            previous_point = None

    cv2.imshow("AirCanvas", display_canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    if key == ord("s"):
        cv2.imwrite("air_drawing.png", canvas)
        print("PNG kaydedildi.")

cap.release()
cv2.destroyAllWindows()
