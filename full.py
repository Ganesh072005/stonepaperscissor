import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import time

class FlyingObjectDetector:
    """
    Real-time flying object detection system using YOLOv8
    Detects birds, airplanes, drones, kites, and other aerial objects
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Flying object classes from COCO dataset
        self.flying_classes = {
            14: 'bird',
            4: 'airplane',
            23: 'bear',  # Sometimes misclassified flying objects
            38: 'kite'
        }
        
        # Track detection history
        self.detection_history = defaultdict(list)
        self.frame_count = 0
        
        # Colors for bounding boxes (BGR format)
        self.colors = {
            'bird': (0, 255, 0),      # Green
            'airplane': (255, 0, 0),   # Blue
            'kite': (0, 165, 255),     # Orange
            'other': (0, 255, 255)     # Yellow
        }
    
    def detect_frame(self, frame):
        """
        Detect flying objects in a single frame
        
        Args:
            frame: Input image frame
            
        Returns:
            annotated_frame: Frame with bounding boxes
            detections: List of detection dictionaries
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Check if detected object is a flying object
                if cls_id in self.flying_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = self.flying_classes[cls_id]
                    
                    # Store detection
                    detection = {
                        'frame': self.frame_count,
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    color = self.colors.get(class_name, self.colors['other'])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with background
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.frame_count += 1
        return annotated_frame, detections
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video file and detect flying objects
        
        Args:
            video_path: Path to input video (or 0 for webcam)
            output_path: Path to save output video
            display: Whether to display results in real-time
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            annotated_frame, detections = self.detect_frame(frame)
            
            # Store detections
            for det in detections:
                self.detection_history[det['class']].append(det)
            
            # Add statistics overlay
            self._add_statistics_overlay(annotated_frame)
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
            
            # Display
            if display:
                cv2.imshow('Flying Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        self.print_summary()
    
    def _add_statistics_overlay(self, frame):
        """Add detection statistics to frame"""
        y_offset = 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for class_name, color in self.colors.items():
            if class_name != 'other':
                count = len([d for d in self.detection_history[class_name] 
                           if d['frame'] == self.frame_count])
                y_offset += 30
                cv2.putText(frame, f"{class_name.capitalize()}: {count}", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def print_summary(self):
        """Print detection summary statistics"""
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        
        for class_name in self.flying_classes.values():
            count = len(self.detection_history[class_name])
            if count > 0:
                avg_conf = np.mean([d['confidence'] for d in self.detection_history[class_name]])
                print(f"{class_name.capitalize()}: {count} detections (avg conf: {avg_conf:.2f})")
        
        print(f"\nTotal frames processed: {self.frame_count}")
    
    def visualize_statistics(self):
        """Create visualization plots of detection statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flying Object Detection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Detection counts by class
        class_counts = {name: len(self.detection_history[name]) 
                       for name in self.flying_classes.values()}
        class_counts = {k: v for k, v in class_counts.items() if v > 0}
        
        if class_counts:
            axes[0, 0].bar(class_counts.keys(), class_counts.values(), 
                          color=['green', 'blue', 'orange', 'yellow'][:len(class_counts)])
            axes[0, 0].set_title('Total Detections by Class')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence distribution
        all_confidences = []
        for detections in self.detection_history.values():
            all_confidences.extend([d['confidence'] for d in detections])
        
        if all_confidences:
            axes[0, 1].hist(all_confidences, bins=20, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Detections over time
        frame_detections = defaultdict(int)
        for detections in self.detection_history.values():
            for det in detections:
                frame_detections[det['frame']] += 1
        
        if frame_detections:
            frames = sorted(frame_detections.keys())
            counts = [frame_detections[f] for f in frames]
            axes[1, 0].plot(frames, counts, color='purple', linewidth=2)
            axes[1, 0].set_title('Detections Over Time')
            axes[1, 0].set_xlabel('Frame Number')
            axes[1, 0].set_ylabel('Number of Detections')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Average confidence by class
        avg_conf = {}
        for class_name in self.flying_classes.values():
            if self.detection_history[class_name]:
                avg_conf[class_name] = np.mean([d['confidence'] 
                                                for d in self.detection_history[class_name]])
        
        if avg_conf:
            axes[1, 1].bar(avg_conf.keys(), avg_conf.values(), 
                          color=['green', 'blue', 'orange', 'yellow'][:len(avg_conf)])
            axes[1, 1].set_title('Average Confidence by Class')
            axes[1, 1].set_ylabel('Confidence')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('detection_analysis.png', dpi=300, bbox_inches='tight')
        print("\nAnalysis plots saved as 'detection_analysis.png'")
        plt.show()
    
    def export_results_to_csv(self, filename='detections.csv'):
        """Export detection results to CSV file"""
        all_detections = []
        
        for class_name, detections in self.detection_history.items():
            for det in detections:
                all_detections.append({
                    'frame': det['frame'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'x1': det['bbox'][0],
                    'y1': det['bbox'][1],
                    'x2': det['bbox'][2],
                    'y2': det['bbox'][3],
                    'center_x': det['center'][0],
                    'center_y': det['center'][1]
                })
        
        df = pd.DataFrame(all_detections)
        df.to_csv(filename, index=False)
        print(f"\nResults exported to '{filename}'")
        return df


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FlyingObjectDetector(
        model_path='yolov8n.pt',  # Use 'yolov8s.pt', 'yolov8m.pt' for better accuracy
        confidence_threshold=0.5
    )
    
    # Option 1: Process video file
    # detector.process_video('input_video.mp4', output_path='output_video.mp4', display=True)
    
    # Option 2: Real-time webcam detection
    # detector.process_video(0, display=True)
    
    # Option 3: Process and analyze
    print("Flying Object Detection System")
    print("="*50)
    print("\nOptions:")
    print("1. Process video file")
    print("2. Real-time webcam detection")
    print("3. Process image")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == '1':
        video_path = input("Enter video file path: ")
        save_output = input("Save output video? (y/n): ").lower() == 'y'
        output_path = 'flying_objects_output.mp4' if save_output else None
        
        detector.process_video(video_path, output_path=output_path, display=True)
        detector.visualize_statistics()
        detector.export_results_to_csv()
        
    elif choice == '2':
        print("\nStarting webcam... Press 'q' to quit")
        detector.process_video(0, display=True)
        detector.visualize_statistics()
        
    elif choice == '3':
    image_path = input("Enter image file path: ").strip()

    # ✅ Check if file exists before trying to read it
    import os
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        print("Hint: Make sure your image is extracted from ZIP and the path is correct.")
    else:
        img = cv2.imread(image_path)

        if img is None:
            print("❌ Error: Could not read image. Check the file format or permissions.")
        else:
            annotated, detections = detector.detect_frame(img)
            cv2.imshow('Detection Result', annotated)
            cv2.imwrite('detected_output.jpg', annotated)
            print(f"\n✅ Detected {len(detections)} flying object(s)")
            print("📁 Output saved as 'detected_output.jpg'")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    
    else:
        print("Invalid option")
        print("Invalid option")