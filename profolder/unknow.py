"""
Flying Object Detection using YOLOv8

Description:
    - Detects flying objects (aircraft, drones, birds, helicopter)
    - Uses trained YOLOv8 model (.pt)
    - Runs entirely on CPU (compatible with Windows)
    - Supports 3 inputs: Video file, DroidCam client, Laptop webcam
    - Supports TRUE fullscreen toggle (F key)
"""

import cv2
from ultralytics import YOLO
import os
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\weights\generalized_40_class\best.pt"
VIDEO_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\bird.mp4"
CONFIDENCE = 0.4
DEVICE = "cpu"

# -------------------------------
# INPUT SELECTION
# -------------------------------
print("\nSELECT INPUT SOURCE:")
print("1 → Video File")
print("2 → DroidCam Client")
print("3 → Laptop Webcam")
choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    INPUT_SOURCE = "video"
elif choice == "2":
    INPUT_SOURCE = "droidcam"
elif choice == "3":
    INPUT_SOURCE = "webcam"
else:
    INPUT_SOURCE = "video"

# -------------------------------
# LOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found")

model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# -------------------------------
# OPEN INPUT SOURCE
# -------------------------------
if INPUT_SOURCE == "video":
    cap = cv2.VideoCapture(VIDEO_PATH)
elif INPUT_SOURCE == "droidcam":
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(2)
elif INPUT_SOURCE == "webcam":
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open input source")

# -------------------------------
# FPS CALCULATION
# -------------------------------
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps if fps > 0 else 0.04

# -------------------------------
# WINDOW SETUP (REAL FULLSCREEN)
# -------------------------------
WINDOW_NAME = "Flying Object Detection (YOLOv8 - CPU)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

fullscreen = False
cv2.resizeWindow(WINDOW_NAME, 960, 540)

# -------------------------------
# DETECTION LOOP
# -------------------------------
previous_positions = {}
object_speeds = {}

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if label in previous_positions:
                dx = cx - previous_positions[label][0]
                dy = cy - previous_positions[label][1]
                dist = (dx * dx + dy * dy) ** 0.5
                object_speeds[label] = dist / frame_time
            else:
                object_speeds[label] = 0

            previous_positions[label] = (cx, cy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"class:{label} :: {object_speeds[label]:.1f}px/s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Toggle fullscreen
    if key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )
            cv2.resizeWindow(WINDOW_NAME, 960, 540)

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()

end_time = time.time()
print(f"Detection completed in {end_time - start_time:.2f} seconds")
