"""
FAST Flying Object Detection using YOLOv8 (Optimized for CPU)

Description:
    - Detects flying objects (aircraft, drones, birds, balloons)
    - Uses YOLOv8 (small or nano model for faster performance)
    - Processes video efficiently by resizing + skipping frames
    - Displays real-time detection with speed estimation
"""

import cv2
from ultralytics import YOLO
import os
import time

# ===================================
# CONFIGURATION SECTION
# ===================================

MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\weights\generalized_40_class\best.pt"
VIDEO_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\bird.mp4"

CONFIDENCE = 0.35   # Lower for faster + more detections
DEVICE = 'cpu'       # Use 'cuda' if you have GPU

FRAME_SKIP = 2       # Process every 2nd frame (increase for more speed)
RESIZE_WIDTH = 640   # Reduce frame size for faster detection
RESIZE_HEIGHT = 360

# ===================================
# INITIALIZATION
# ===================================

print("\n🚀 Initializing Fast Flying Object Detection...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"❌ Video not found at: {VIDEO_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"❌ Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps if fps > 0 else 0.04
print(f"🎥 Video loaded ({fps:.2f} FPS). Starting optimized detection...\n")

# ===================================
# DETECTION LOOP
# ===================================

previous_positions = {}
object_speeds = {}
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video processing complete.")
        break

    frame_count += 1
    # Skip frames for speed boost
    if frame_count % FRAME_SKIP != 0:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Run detection
    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    # Loop through detections
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Speed calculation
            if label in previous_positions:
                dx = cx - previous_positions[label][0]
                dy = cy - previous_positions[label][1]
                distance = (dx**2 + dy**2)**0.5
                object_speeds[label] = distance / (frame_time * FRAME_SKIP)
            else:
                object_speeds[label] = 0

            previous_positions[label] = (cx, cy)

            # Draw detection box and speed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label}: {object_speeds[label]:.1f}px/s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 600, 500),
                2
            )

    # Display the frame
    cv2.imshow("🚀 Fast Flying Object Detection (YOLOv8 - CPU)", frame)

    # Quit with 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Detection stopped by user.")
        break

# ===================================
# CLEANUP
# ===================================

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
print(f"\n✅ Detection completed in {end_time - start_time:.2f} seconds.")
print("📂 Output video frames (if saved) will appear in: 'runs/detect/predict'")
print("⚡ FPS improved with frame skipping and resizing.")
