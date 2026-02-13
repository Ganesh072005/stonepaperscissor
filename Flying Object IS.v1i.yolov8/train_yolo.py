# train_yolo.py

from ultralytics import YOLO

# Load the YOLOv8 model (you can use yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")  # 'n' = nano model for faster training on smaller datasets

# Path to your dataset YAML file
data_path = r"C:\Users\harsh\OneDrive\Desktop\miniproject\Flying Object IS.v1i.yolov8\data.yaml"

# Start training
model.train(
    data=data_path,      # Path to data.yaml
    epochs=50,           # Number of training rounds
    imgsz=640,           # Image size (default 640)
    batch=8,             # Number of images per batch (adjust if low memory)
    name="flying_object_v1",  # Folder name for results
    project=r"C:\Users\harsh\OneDrive\Desktop\miniproject\runs",  # Output directory
)

print(" Training complete! Check the 'runs/detect/flying_object_v1' folder for results.")
