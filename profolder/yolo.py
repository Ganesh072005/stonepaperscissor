from ultralytics import YOLO
import torch

# Load trained model
model = YOLO(r'C:\Users\harsh\OneDrive\Desktop\miniproject\Real-Time-Flying-Object-Detection_with_YOLOv8-main\weights\best.pt')

# 0 for GPU, 'cpu' for CPU
device = 0  

# Confidence threshold
conf = 0.67

# Source (image path, video path, or URL)
source = r'C:\Users\harsh\Downloads\archive.zip\inputData\img-vid\img-bird3.jpg'

# Run detection
model.predict(source=source, save=True, conf=conf, device=device, verbose=False)
