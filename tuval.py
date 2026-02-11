"""
Ultimate Air Gesture Drawing System
-----------------------------------
Features:
- True fullscreen canvas (no top gap)
- Rock gesture eraser (index + pinky)
- Thickness control (thumb-index distance)
- Color palette
- FPS stabilization
- Smoothing
- Small camera preview (top-right overlay)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math


# ============================
# CONFIGURATION
# ============================

TARGET_FPS = 60
SMOOTHING_FACTOR = 0.7

BRUSH_MIN = 3
BRUSH_MAX = 40
ERASER_SIZE = 60

MIN_DETECTION_CONF = 0.8
MIN_TRACKING_CONF = 0.8


# ============================
# INITIALIZATION
# ============================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
    max_num_hands=2
)

cap = cv2.VideoCapture(0)

success, frame_init = cap.read()
height, width, _ = frame_init.shape

cv2.namedWindow("AirCanvas", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("AirCanvas",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

canvas = np.ones((height, width, 3), dtype=np.uint8) * 255


# ============================
# STATE VARIABLES
# ============================

previous_points = {}
smoothed_points = {}

draw_color = (0, 0, 255)
brush_thickness = 8

color_palette = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255)
]

selected_color_index = 0

previous_time = time.time()


# ============================
# HELPER FUNCTIONS
# ============================

def smooth_point(prev, current):
    if prev is None:
        return current
    x = int(prev[0] * SMOOTHING_FACTOR + current[0] * (1 - SMOOTHING_FACTOR))
    y = int(prev[1] * SMOOTHING_FACTOR + current[1] * (1 - SMOOTHING_FACTOR))
    return (x, y)


def is_finger_up(tip, dip):
    return tip[1] < dip[1]


def calculate_distance(p1, p2):
    return int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def map_thickness(distance):
    distance = max(20, min(distance, 200))
    return int(np.interp(distance,
                         [20, 200],
                         [BRUSH_MIN, BRUSH_MAX]))


# ============================
# MAIN LOOP
# ============================

while cap.isOpened():

    # FPS stabilization
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

    # Tuval kopyası
    display_canvas = canvas.copy()

    # ============================
    # COLOR PALETTE (TOP BAR)
    # ============================

    for i, color in enumerate(color_palette):
        x_pos = 50 + i * 80
        cv2.rectangle(display_canvas,
                      (x_pos, 20),
                      (x_pos + 60, 80),
                      color,
                      cv2.FILLED)

        if i == selected_color_index:
            cv2.rectangle(display_canvas,
                          (x_pos, 20),
                          (x_pos + 60, 80),
                          (0, 0, 0),
                          3)

    # ============================
    # SMALL CAMERA PREVIEW
    # ============================

    preview_w = int(width * 0.25)
    preview_h = int(height * 0.25)

    small_frame = cv2.resize(frame, (preview_w, preview_h))

    display_canvas[0:preview_h,
                   width - preview_w:width] = small_frame

    # ============================
    # FPS TEXT
    # ============================

    fps = int(1 / max(elapsed, 0.0001))

    cv2.putText(display_canvas,
                f"FPS: {fps}",
                (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (50, 50, 50),
                2)

    # ============================
    # HAND PROCESSING
    # ============================

    if results.multi_hand_landmarks:

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            points = []

            for lm in hand_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
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

            # Smoothing
            prev_smooth = smoothed_points.get(hand_index)
            smooth = smooth_point(prev_smooth, index_tip)
            smoothed_points[hand_index] = smooth

            cv2.circle(display_canvas, smooth, 8, (0, 0, 0), cv2.FILLED)

            # ============================
            # THICKNESS CONTROL
            # Thumb + Index mesafesi
            # ============================

            thumb_distance = calculate_distance(thumb_tip, index_tip)
            brush_thickness = map_thickness(thumb_distance)

            # ============================
            # DRAW (Only Index)
            # ============================

            if index_up and not middle_up and not ring_up and not pinky_up:

                # Renk seçimi
                for i in range(len(color_palette)):
                    x_start = 50 + i * 80
                    if x_start < smooth[0] < x_start + 60 and 20 < smooth[1] < 80:
                        selected_color_index = i
                        draw_color = color_palette[i]

                prev = previous_points.get(hand_index)

                if prev is None:
                    previous_points[hand_index] = smooth
                else:
                    cv2.line(canvas,
                             prev,
                             smooth,
                             draw_color,
                             brush_thickness)
                    previous_points[hand_index] = smooth

            # ============================
            # ERASE (ROCK 🤘)
            # ============================

            elif index_up and pinky_up and not middle_up and not ring_up:

                previous_points[hand_index] = None

                cv2.circle(canvas,
                           smooth,
                           ERASER_SIZE,
                           (255, 255, 255),
                           cv2.FILLED)

            # ============================
            # INDEX + MIDDLE → DO NOTHING
            # ============================

            else:
                previous_points[hand_index] = None

    cv2.imshow("AirCanvas", display_canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255


cap.release()
cv2.destroyAllWindows()
