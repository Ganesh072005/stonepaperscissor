"""
Flying Object Detection using YOLOv8

Description:
    - Detects flying objects (aircraft, drones, birds, helicopter)
    - Uses trained YOLOv8 model (.pt)
    - Runs entirely on CPU (compatible with Windows)
    - Supports 3 inputs: Video file, DroidCam client, Laptop webcam
"""

import cv2
from ultralytics import YOLO
import os
import time

# -------------------------------
#  CONFIGURATION SECTION
# -------------------------------

MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\weights\generalized_40_class\best.pt"

# Default video
VIDEO_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\bird.mp4"

# Confidence threshold
CONFIDENCE = 0.4

# Device
DEVICE = 'cpu'

# -------------------------------
# INPUT SELECTION (ADD-ON)
# -------------------------------
print("\nSELECT INPUT SOURCE:")
print("1 → Video File")
print("2 → DroidCam Client (Windows)")
print("3 → Laptop Webcam")
choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    INPUT_SOURCE = "video"

elif choice == "2":
    INPUT_SOURCE = "droidcam"

elif choice == "3":
    INPUT_SOURCE = "webcam"

else:
    print("Invalid choice! Defaulting to video file.")
    INPUT_SOURCE = "video"

# -------------------------------
# INITIALIZATION (UNCHANGED)
# -------------------------------

print("\nInitializing Flying Object Detection...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# -------------------------------
# OPEN CORRECT INPUT SOURCE
# -------------------------------

if INPUT_SOURCE == "video":
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found at: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Input: Video File")

elif INPUT_SOURCE == "droidcam":
    print("Input: DroidCam Client Feed")

    # Try index 1 → then index 2
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Index 1 failed → trying index 2...")
        cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        raise IOError("DroidCam NOT detected. Open DroidCam Client and try again.")

elif INPUT_SOURCE == "webcam":
    print("Input: Laptop Webcam (index 0)")
    cap = cv2.VideoCapture(0)

else:
    raise RuntimeError("Unknown input source!")

# Check camera open
if not cap.isOpened():
    raise IOError("Could not open selected input source!")

# -------------------------------
# FPS FOR SPEED CALCULATION
# -------------------------------
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps if fps > 0 else 0.04

print(f"Stream FPS: {fps:.2f} (if 0, webcam FPS is dynamic)")
print("Starting detection...")

# -------------------------------
# DETECTION LOOP (UNCHANGED)
# -------------------------------

previous_positions = {}
object_speeds = {}

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
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
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # Speed calculation (unchanged)
            if label in previous_positions:
                dx = cx - previous_positions[label][0]
                dy = cy - previous_positions[label][1]
                distance_pixels = (dx*dx + dy*dy)**0.5
                object_speeds[label] = distance_pixels / (frame_time *20)
            else:
                object_speeds[label] = 0

            previous_positions[label] = (cx, cy)

            # Draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"class:{label} :: {object_speeds[label]:.1f}px/s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )

    cv2.imshow("Flying Object Detection (YOLOv8 - CPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detection stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
print(f"\nDetection completed successfully in {end_time - start_time:.2f} seconds.")
print("Output video frames saved in: 'runs/detect/predict'")
