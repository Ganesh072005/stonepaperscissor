"""
Flying Object Speed Detection (Webcam Version)
Author: Harsha M G
Description:
    - Runs YOLOv8 object detection from live webcam feed
    - Tracks motion and computes approximate speed
    - Works 100% on CPU (no GPU required)
"""

import cv2
import time
import math
from collections import deque
from ultralytics import YOLO

# ======================
# CONFIGURATION
# ======================

# Path to your trained YOLOv8 model
MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\gitfolder\weights\generalized_40_class\last.pt"

# Use webcam (0 = default camera)
SOURCE = 0

# Confidence threshold for detections
CONFIDENCE = 0.4

# Device (CPU-only)
DEVICE = "cpu"

# Speed calibration (1 meter ≈ X pixels)
PIXELS_PER_METER = 40.0  # tune if you want speed in km/h
SHOW_KMPH = True

# ======================
# SMOOTHING FUNCTION (fixed version)
# ======================
def exponential_smooth(prev_val, new_val, alpha=0.7):
    """
    Smooths either scalars or (x, y) tuples using exponential moving average.
    """
    if prev_val is None:
        return new_val

    if isinstance(prev_val, (tuple, list)) and isinstance(new_val, (tuple, list)):
        return (
            alpha * prev_val[0] + (1 - alpha) * new_val[0],
            alpha * prev_val[1] + (1 - alpha) * new_val[1],
        )
    else:
        return alpha * prev_val + (1 - alpha) * new_val


# ======================
# MAIN
# ======================

print("\n🚀 Initializing YOLOv8 for Real-Time Detection...")

# Load model
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise IOError("❌ Could not access the webcam. Try a different index (1 or 2).")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_time = 1.0 / fps

print(f"🎥 Webcam started at {fps:.1f} FPS.")

# Dictionaries to store object data
previous_positions = {}
object_speeds = {}
smoothed_positions = {}
track_history = {}

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("🛑 Webcam feed ended.")
        break

    # YOLO inference
    results = model.predict(source=frame, conf=CONFIDENCE, device=DEVICE, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Smooth the center position
            smoothed_positions[label] = exponential_smooth(
                smoothed_positions.get(label), (cx, cy)
            )

            # Compute speed (in pixels/second)
            if label in previous_positions:
                dx = smoothed_positions[label][0] - previous_positions[label][0]
                dy = smoothed_positions[label][1] - previous_positions[label][1]
                distance_px = math.sqrt(dx**2 + dy**2)
                speed_px_s = distance_px / frame_time
                object_speeds[label] = exponential_smooth(
                    object_speeds.get(label, 0), speed_px_s
                )
            else:
                object_speeds[label] = 0.0

            previous_positions[label] = smoothed_positions[label]

            # Convert to km/h (optional)
            if PIXELS_PER_METER:
                speed_mps = object_speeds[label] / PIXELS_PER_METER
                speed_kmph = speed_mps * 3.6*100
            else:
                speed_kmph = object_speeds[label]

            # Store trail history for nice visual effect
            if label not in track_history:
                track_history[label] = deque(maxlen=25)
            track_history[label].append(smoothed_positions[label])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

            # Draw motion trail
            for i in range(1, len(track_history[label])):
                cv2.line(
                    frame,
                    (int(track_history[label][i - 1][0]), int(track_history[label][i - 1][1])),
                    (int(track_history[label][i][0]), int(track_history[label][i][1])),
                    (255, 200, 60),
                    2,
                )

            # Draw label and speed
            text = f"{label}: {speed_kmph:.1f} km/h" if SHOW_KMPH else f"{label}: {object_speeds[label]:.1f}px/s"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("🛰️ Flying Object Speed Detection (YOLOv8 - CPU)", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Session complete in {time.time() - start_time:.2f} seconds.")
