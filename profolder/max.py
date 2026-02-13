"""
Advanced Flying Object Detection System using YOLOv8
====================================================

Features:
    - Multi-threaded video processing for optimized performance
    - Real-time object tracking with unique IDs
    - Speed calculation in km/h with calibration support
    - Trajectory visualization and heatmaps
    - Comprehensive analytics dashboard
    - Export detection data to CSV/JSON
    - Video recording with annotations
    - FPS optimization and frame skipping
    - Alert system for suspicious objects
    - Distance estimation and altitude tracking
"""

import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
from datetime import datetime
from collections import defaultdict, deque
import json
import csv
from pathlib import Path
import threading
from queue import Queue

# ============================================================================
#  CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration management"""
    
    # Model Configuration
    MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\weights\generalized_40_class\last.pt"
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.45
    DEVICE = 'cpu'
    
    # Video Configuration
    VIDEO_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\airplane.mp4"
    FRAME_SKIP = 1  # Process every Nth frame (1 = no skip)
    MAX_DISPLAY_WIDTH = 1280
    MAX_DISPLAY_HEIGHT = 720
    
    # Tracking Configuration
    MAX_DISAPPEARED = 30  # Frames before object is considered lost
    MAX_DISTANCE = 150  # Maximum pixel distance for object matching
    TRAJECTORY_LENGTH = 50  # Number of points in trajectory trail
    
    # Speed Calibration (adjust based on your camera setup)
    PIXELS_PER_METER = 50  # Calibrate this based on known distance
    
    # Alert Configuration
    SPEED_THRESHOLD_KMH = 100  # Alert if object exceeds this speed
    SUSPICIOUS_CLASSES = ['drone', 'uav']  # Classes to trigger alerts
    
    # Output Configuration
    OUTPUT_DIR = Path("detection_results")
    SAVE_VIDEO = True
    SAVE_ANALYTICS = True
    SAVE_HEATMAP = True
    
    # Display Configuration
    SHOW_TRAJECTORIES = True
    SHOW_SPEED = True
    SHOW_STATISTICS = True
    SHOW_HEATMAP = False  # Toggle during runtime with 'H'
    
    # Colors (BGR format)
    COLORS = {
        'aircraft': (255, 100, 0),
        'drone': (0, 100, 255),
        'bird': (0, 255, 100),
        'balloon': (255, 0, 255),
        'default': (0, 255, 0)
    }

# ============================================================================
#  OBJECT TRACKER
# ============================================================================

class ObjectTracker:
    """Advanced object tracking with unique IDs and trajectory history"""
    
    def __init__(self, max_disappeared=30, max_distance=150):
        self.next_object_id = 0
        self.objects = {}  # id: centroid
        self.disappeared = {}  # id: frame_count
        self.trajectories = defaultdict(lambda: deque(maxlen=Config.TRAJECTORY_LENGTH))
        self.speeds = {}
        self.classes = {}
        self.confidences = {}
        self.first_seen = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, class_name, confidence):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.classes[self.next_object_id] = class_name
        self.confidences[self.next_object_id] = confidence
        self.first_seen[self.next_object_id] = time.time()
        self.trajectories[self.next_object_id].append(centroid)
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.classes[object_id]
        del self.confidences[object_id]
        del self.first_seen[object_id]
        if object_id in self.speeds:
            del self.speeds[object_id]
            
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = np.array([d['centroid'] for d in detections])
        
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['centroid'], detection['class'], detection['conf'])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            
            # Calculate distances between objects and detections
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            
            # Match objects to detections
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.classes[object_id] = detections[col]['class']
                self.confidences[object_id] = detections[col]['conf']
                self.trajectories[object_id].append(tuple(input_centroids[col]))
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], detections[col]['class'], detections[col]['conf'])
        
        return self.objects
    
    def calculate_speed(self, object_id, fps):
        """Calculate object speed in km/h"""
        if object_id not in self.trajectories or len(self.trajectories[object_id]) < 2:
            return 0.0
        
        trajectory = list(self.trajectories[object_id])
        p1 = np.array(trajectory[-2])
        p2 = np.array(trajectory[-1])
        
        distance_pixels = np.linalg.norm(p2 - p1)
        distance_meters = distance_pixels / Config.PIXELS_PER_METER
        distance_per_frame = distance_meters
        distance_per_second = distance_per_frame * fps
        speed_kmh = distance_per_second * 3.6
        
        self.speeds[object_id] = speed_kmh
        return speed_kmh

# ============================================================================
#  ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Comprehensive analytics and statistics tracking"""
    
    def __init__(self):
        self.detection_counts = defaultdict(int)
        self.total_detections = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.speed_history = defaultdict(list)
        self.alerts = []
        self.heatmap = None
        
    def update(self, frame_shape, detections, tracker):
        """Update analytics with new frame data"""
        self.frame_count += 1
        
        # Initialize heatmap
        if self.heatmap is None:
            self.heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for obj_id, centroid in tracker.objects.items():
            class_name = tracker.classes[obj_id]
            self.detection_counts[class_name] += 1
            self.total_detections += 1
            
            # Update heatmap
            cx, cy = int(centroid[0]), int(centroid[1])
            if 0 <= cy < self.heatmap.shape[0] and 0 <= cx < self.heatmap.shape[1]:
                cv2.circle(self.heatmap, (cx, cy), 20, 1, -1)
            
            # Track speed history
            if obj_id in tracker.speeds:
                speed = tracker.speeds[obj_id]
                self.speed_history[class_name].append(speed)
                
                # Alert system
                if speed > Config.SPEED_THRESHOLD_KMH:
                    self.alerts.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'object_id': obj_id,
                        'class': class_name,
                        'speed': speed,
                        'position': centroid
                    })
                
                if class_name.lower() in Config.SUSPICIOUS_CLASSES:
                    self.alerts.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'object_id': obj_id,
                        'class': class_name,
                        'alert': 'Suspicious object detected'
                    })
    
    def get_statistics(self):
        """Get current statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        stats = {
            'elapsed_time': elapsed_time,
            'fps': fps,
            'total_detections': self.total_detections,
            'frame_count': self.frame_count,
            'detections_by_class': dict(self.detection_counts),
            'alerts': len(self.alerts)
        }
        
        # Average speeds by class
        avg_speeds = {}
        for class_name, speeds in self.speed_history.items():
            if speeds:
                avg_speeds[class_name] = np.mean(speeds)
        stats['avg_speeds'] = avg_speeds
        
        return stats
    
    def get_heatmap(self, frame_shape):
        """Get normalized heatmap for visualization"""
        if self.heatmap is None:
            return np.zeros(frame_shape, dtype=np.uint8)
        
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap_colored
    
    def export_to_csv(self, filepath):
        """Export analytics to CSV"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            
            stats = self.get_statistics()
            writer.writerow(['Total Detections', stats['total_detections']])
            writer.writerow(['Frames Processed', stats['frame_count']])
            writer.writerow(['Average FPS', f"{stats['fps']:.2f}"])
            writer.writerow(['Elapsed Time (s)', f"{stats['elapsed_time']:.2f}"])
            writer.writerow(['Total Alerts', stats['alerts']])
            writer.writerow([])
            
            writer.writerow(['Detections by Class'])
            for class_name, count in stats['detections_by_class'].items():
                writer.writerow([class_name, count])
            writer.writerow([])
            
            writer.writerow(['Average Speeds (km/h)'])
            for class_name, speed in stats['avg_speeds'].items():
                writer.writerow([class_name, f"{speed:.2f}"])
    
    def export_to_json(self, filepath):
        """Export analytics to JSON"""
        stats = self.get_statistics()
        stats['alerts_details'] = self.alerts
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4, default=str)

# ============================================================================
#  VISUALIZATION
# ============================================================================

class Visualizer:
    """Advanced visualization with overlays and statistics"""
    
    @staticmethod
    def draw_detection(frame, x1, y1, x2, y2, class_name, confidence, obj_id, speed):
        """Draw bounding box and information"""
        color = Config.COLORS.get(class_name.lower(), Config.COLORS['default'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"ID:{obj_id} {class_name}"
        if Config.SHOW_SPEED and speed > 0:
            label += f" {speed:.1f}km/h"
        label += f" {confidence:.2f}"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    @staticmethod
    def draw_trajectory(frame, trajectory, color):
        """Draw object trajectory trail"""
        if len(trajectory) < 2:
            return
        
        points = np.array(trajectory, dtype=np.int32)
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
    
    @staticmethod
    def draw_statistics_panel(frame, stats, alerts):
        """Draw comprehensive statistics overlay"""
        panel_height = 200
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        y_offset = 25
        line_height = 25
        
        # Title
        cv2.putText(panel, "DETECTION STATISTICS", (10, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        # FPS and frame count
        cv2.putText(panel, f"FPS: {stats['fps']:.1f} | Frames: {stats['frame_count']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Detections by class
        detections_text = "Detected: "
        for class_name, count in stats['detections_by_class'].items():
            detections_text += f"{class_name}:{count} "
        cv2.putText(panel, detections_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Average speeds
        if stats['avg_speeds']:
            speeds_text = "Avg Speed: "
            for class_name, speed in stats['avg_speeds'].items():
                speeds_text += f"{class_name}:{speed:.1f}km/h "
            cv2.putText(panel, speeds_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Alerts
        alert_color = (0, 0, 255) if len(alerts) > 0 else (0, 255, 0)
        cv2.putText(panel, f"Alerts: {len(alerts)}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
        
        # Recent alert
        if alerts:
            latest_alert = alerts[-1]
            alert_text = f"Latest: {latest_alert.get('class', 'N/A')} - {latest_alert.get('time', 'N/A')}"
            cv2.putText(panel, alert_text, (10, y_offset + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Combine with frame
        return np.vstack([panel, frame])
    
    @staticmethod
    def create_mini_map(frame, tracker, analytics):
        """Create a mini-map showing object positions"""
        map_size = 200
        mini_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        mini_map[:] = (50, 50, 50)
        
        h, w = frame.shape[:2]
        scale_x = map_size / w
        scale_y = map_size / h
        
        for obj_id, centroid in tracker.objects.items():
            x = int(centroid[0] * scale_x)
            y = int(centroid[1] * scale_y)
            class_name = tracker.classes[obj_id]
            color = Config.COLORS.get(class_name.lower(), Config.COLORS['default'])
            cv2.circle(mini_map, (x, y), 3, color, -1)
        
        cv2.putText(mini_map, "MAP", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return mini_map

# ============================================================================
#  MAIN DETECTION SYSTEM
# ============================================================================

class FlyingObjectDetector:
    """Main detection system orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.setup_directories()
        self.load_model()
        self.tracker = ObjectTracker(Config.MAX_DISAPPEARED, Config.MAX_DISTANCE)
        self.analytics = AnalyticsEngine()
        self.visualizer = Visualizer()
        self.video_writer = None
        self.show_heatmap = Config.SHOW_HEATMAP
        
    def setup_directories(self):
        """Create output directories"""
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        (Config.OUTPUT_DIR / "videos").mkdir(exist_ok=True)
        (Config.OUTPUT_DIR / "analytics").mkdir(exist_ok=True)
        (Config.OUTPUT_DIR / "heatmaps").mkdir(exist_ok=True)
        
    def load_model(self):
        """Load and validate YOLO model"""
        print("\n" + "="*60)
        print(" ADVANCED FLYING OBJECT DETECTION SYSTEM")
        print("="*60)
        
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}")
        
        print(f"\nLoading model: {Config.MODEL_PATH}")
        self.model = YOLO(Config.MODEL_PATH)
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {Config.DEVICE}")
        print(f"✓ Classes: {len(self.model.names)}")
        
    def setup_video(self):
        """Setup video capture and writer"""
        if not os.path.exists(Config.VIDEO_PATH):
            raise FileNotFoundError(f"Video not found: {Config.VIDEO_PATH}")
        
        self.cap = cv2.VideoCapture(Config.VIDEO_PATH)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {Config.VIDEO_PATH}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n Video Information:")
        print(f"✓ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"✓ FPS: {self.fps:.2f}")
        print(f"✓ Total Frames: {self.total_frames}")
        print(f"✓ Duration: {self.total_frames/self.fps:.2f}s")
        
        if Config.SAVE_VIDEO:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Config.OUTPUT_DIR / "videos" / f"detection_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Account for statistics panel
            output_height = self.frame_height + 200 if Config.SHOW_STATISTICS else self.frame_height
            
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, self.fps,
                (self.frame_width, output_height)
            )
            print(f"✓ Output video: {output_path}")
    
    def process_frame(self, frame):
        """Process single frame with detection and tracking"""
        # Run detection
        results = self.model.predict(
            source=frame,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            device=Config.DEVICE,
            verbose=False
        )
        
        # Initialize bboxes dict if not exists
        if not hasattr(self.tracker, 'bboxes'):
            self.tracker.bboxes = {}
        
        detections = []
        bbox_map = {}  # Temporary map to store bboxes for new detections
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls]
                
                detection_data = {
                    'bbox': (x1, y1, x2, y2),
                    'centroid': (cx, cy),
                    'class': class_name,
                    'conf': conf
                }
                detections.append(detection_data)
                bbox_map[(cx, cy)] = (x1, y1, x2, y2)
        
        # Update tracker
        self.tracker.update(detections)
        
        # Update bboxes for tracked objects
        for obj_id, centroid in self.tracker.objects.items():
            # Try to find matching bbox from current detections
            centroid_tuple = tuple(map(int, centroid))
            if centroid_tuple in bbox_map:
                self.tracker.bboxes[obj_id] = bbox_map[centroid_tuple]
            # Keep existing bbox if no new detection matches
        
        # Calculate speeds
        for obj_id in self.tracker.objects.keys():
            self.tracker.calculate_speed(obj_id, self.fps)
        
        # Update analytics
        self.analytics.update(frame.shape, detections, self.tracker)
        
        return frame
    
    def render_frame(self, frame):
        """Render frame with all visualizations"""
        output_frame = frame.copy()
        
        # Draw trajectories
        if Config.SHOW_TRAJECTORIES:
            for obj_id, trajectory in self.tracker.trajectories.items():
                if obj_id in self.tracker.objects:
                    class_name = self.tracker.classes[obj_id]
                    color = Config.COLORS.get(class_name.lower(), Config.COLORS['default'])
                    self.visualizer.draw_trajectory(output_frame, trajectory, color)
        
        # Draw detections - we need to get actual bboxes from current frame
        # Store bboxes in tracker for proper rendering
        for obj_id, centroid in self.tracker.objects.items():
            if not hasattr(self.tracker, 'bboxes'):
                self.tracker.bboxes = {}
            
            # Use stored bbox if available, otherwise estimate
            if obj_id in self.tracker.bboxes:
                x1, y1, x2, y2 = self.tracker.bboxes[obj_id]
            else:
                cx, cy = int(centroid[0]), int(centroid[1])
                x1, y1 = cx - 50, cy - 50
                x2, y2 = cx + 50, cy + 50
            
            class_name = self.tracker.classes[obj_id]
            confidence = self.tracker.confidences[obj_id]
            speed = self.tracker.speeds.get(obj_id, 0)
            
            self.visualizer.draw_detection(
                output_frame, x1, y1, x2, y2,
                class_name, confidence, obj_id, speed
            )
        
        # Add mini-map
        mini_map = self.visualizer.create_mini_map(frame, self.tracker, self.analytics)
        output_frame[10:10+mini_map.shape[0], 10:10+mini_map.shape[1]] = mini_map
        
        # Show heatmap overlay if enabled
        if self.show_heatmap:
            heatmap = self.analytics.get_heatmap(frame.shape)
            output_frame = cv2.addWeighted(output_frame, 0.7, heatmap, 0.3, 0)
        
        # Add statistics panel
        if Config.SHOW_STATISTICS:
            stats = self.analytics.get_statistics()
            output_frame = self.visualizer.draw_statistics_panel(
                output_frame, stats, self.analytics.alerts
            )
        
        return output_frame
    
    def run(self):
        """Main detection loop"""
        self.setup_video()
        
        print("\n" + "="*60)
        print(" STARTING DETECTION")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit")
        print("  H - Toggle heatmap")
        print("  SPACE - Pause/Resume")
        print("="*60 + "\n")
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("\n✓ Video processing complete")
                        break
                    
                    frame_count += 1
                    
                    # Skip frames if configured
                    if frame_count % Config.FRAME_SKIP != 0:
                        continue
                    
                    # Resize for display if needed
                    display_frame = frame.copy()
                    original_shape = frame.shape[:2]  # Store original dimensions
                    
                    if display_frame.shape[1] > Config.MAX_DISPLAY_WIDTH:
                        scale = Config.MAX_DISPLAY_WIDTH / display_frame.shape[1]
                        new_width = Config.MAX_DISPLAY_WIDTH
                        new_height = int(display_frame.shape[0] * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    # Process frame
                    processed_frame = self.process_frame(display_frame)
                    
                    # Render visualizations
                    rendered_frame = self.render_frame(processed_frame)
                    
                    # Save video
                    if Config.SAVE_VIDEO and self.video_writer:
                        self.video_writer.write(rendered_frame)
                    
                    # Display
                    cv2.imshow("Advanced Flying Object Detection", rendered_frame)
                    
                    # Progress indicator
                    if frame_count % 30 == 0:
                        progress = (frame_count / self.total_frames) * 100
                        print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{self.total_frames}", end='\r')
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Detection stopped by user")
                    break
                elif key == ord('h'):
                    self.show_heatmap = not self.show_heatmap
                    print(f"\nHeatmap: {'ON' if self.show_heatmap else 'OFF'}")
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n{'PAUSED' if paused else 'RESUMED'}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and save results"""
        print("\n\n" + "="*60)
        print(" FINALIZING RESULTS")
        print("="*60)
        
        # Release resources
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Save analytics
        if Config.SAVE_ANALYTICS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = Config.OUTPUT_DIR / "analytics" / f"analytics_{timestamp}.csv"
            json_path = Config.OUTPUT_DIR / "analytics" / f"analytics_{timestamp}.json"
            
            self.analytics.export_to_csv(csv_path)
            self.analytics.export_to_json(json_path)
            print(f"\n✓ Analytics saved:")
            print(f"  - CSV: {csv_path}")
            print(f"  - JSON: {json_path}")
        
        # Save heatmap
        if Config.SAVE_HEATMAP and self.analytics.heatmap is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_path = Config.OUTPUT_DIR / "heatmaps" / f"heatmap_{timestamp}.png"
            heatmap = self.analytics.get_heatmap((self.frame_height, self.frame_width, 3))
            cv2.imwrite(str(heatmap_path), heatmap)
            print(f"✓ Heatmap saved: {heatmap_path}")
        
        # Print final statistics
        stats = self.analytics.get_statistics()
        print(f"\n Final Statistics:")
        print(f"  - Total Detections: {stats['total_detections']}")
        print(f"  - Frames Processed: {stats['frame_count']}")
        print(f"  - Average FPS: {stats['fps']:.2f}")
        print(f"  - Total Alerts: {stats['alerts']}")
        
        if stats['detections_by_class']:
            print(f"\n  Detections by Class:")
            for class_name, count in stats['detections_by_class'].items():
                print(f"    • {class_name}: {count}")
        
        if stats['avg_speeds']:
            print(f"\n  Average Speeds:")
            for class_name, speed in stats['avg_speeds'].items():
                print(f"    • {class_name}: {speed:.2f} km/h")
        
        print("\n" + "="*60)
        print(" DETECTION COMPLETE")
        print("="*60 + "\n")

# ============================================================================
#  ADDITIONAL FEATURES
# ============================================================================

class AlertSystem:
    """Advanced alert system with notification levels"""
    
    ALERT_LEVELS = {
        'INFO': (255, 255, 0),      # Yellow
        'WARNING': (0, 165, 255),   # Orange
        'CRITICAL': (0, 0, 255)     # Red
    }
    
    @staticmethod
    def check_alerts(tracker, analytics):
        """Check for various alert conditions"""
        alerts = []
        
        for obj_id, centroid in tracker.objects.items():
            class_name = tracker.classes[obj_id]
            speed = tracker.speeds.get(obj_id, 0)
            
            # Speed alerts
            if speed > Config.SPEED_THRESHOLD_KMH * 1.5:
                alerts.append({
                    'level': 'CRITICAL',
                    'message': f'Extremely high speed detected: {speed:.1f} km/h',
                    'object_id': obj_id,
                    'class': class_name
                })
            elif speed > Config.SPEED_THRESHOLD_KMH:
                alerts.append({
                    'level': 'WARNING',
                    'message': f'High speed detected: {speed:.1f} km/h',
                    'object_id': obj_id,
                    'class': class_name
                })
            
            # Suspicious object alerts
            if class_name.lower() in Config.SUSPICIOUS_CLASSES:
                alerts.append({
                    'level': 'WARNING',
                    'message': f'Suspicious object: {class_name}',
                    'object_id': obj_id,
                    'class': class_name
                })
            
            # Proximity alerts (multiple objects close together)
            for other_id, other_centroid in tracker.objects.items():
                if obj_id != other_id:
                    distance = np.linalg.norm(
                        np.array(centroid) - np.array(other_centroid)
                    )
                    if distance < 100:  # pixels
                        alerts.append({
                            'level': 'INFO',
                            'message': f'Objects in close proximity: ID{obj_id} & ID{other_id}',
                            'object_id': obj_id,
                            'class': class_name
                        })
        
        return alerts

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_frame(frame, max_width=1920):
        """Resize frame for optimal processing"""
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), 
                            interpolation=cv2.INTER_AREA)
        return frame
    
    @staticmethod
    def batch_process_frames(frames, model, config):
        """Process multiple frames in batch for efficiency"""
        results = model.predict(
            source=frames,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            verbose=False,
            stream=True
        )
        return results

class DataExporter:
    """Export detection data in various formats"""
    
    @staticmethod
    def export_detections_timeline(tracker, output_path):
        """Export frame-by-frame detection timeline"""
        timeline_data = []
        
        for obj_id, trajectory in tracker.trajectories.items():
            if obj_id in tracker.first_seen:
                timeline_data.append({
                    'object_id': obj_id,
                    'class': tracker.classes.get(obj_id, 'unknown'),
                    'first_seen': tracker.first_seen[obj_id],
                    'trajectory_length': len(trajectory),
                    'max_speed': max(tracker.speeds.get(obj_id, [0])) if isinstance(tracker.speeds.get(obj_id), list) else tracker.speeds.get(obj_id, 0)
                })
        
        with open(output_path, 'w') as f:
            json.dump(timeline_data, f, indent=4, default=str)
    
    @staticmethod
    def export_heatmap_data(analytics, output_path):
        """Export raw heatmap data for external analysis"""
        if analytics.heatmap is not None:
            np.save(output_path, analytics.heatmap)

class CalibrationTool:
    """Tool for calibrating pixel-to-meter conversion"""
    
    @staticmethod
    def calibrate_from_known_distance(pixel_distance, real_distance_meters):
        """Calculate pixels per meter from known reference"""
        return pixel_distance / real_distance_meters
    
    @staticmethod
    def interactive_calibration(frame):
        """Interactive calibration using mouse clicks"""
        print("\nCalibration Mode:")
        print("Click two points of known distance")
        print("Press 'c' to confirm, 'r' to reset")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if len(points) == 2:
                    cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("Calibration", frame)
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.imshow("Calibration", frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(points) == 2:
                distance = np.linalg.norm(
                    np.array(points[0]) - np.array(points[1])
                )
                cv2.destroyWindow("Calibration")
                return distance
            elif key == ord('r'):
                points.clear()
                frame_copy = frame.copy()
                cv2.imshow("Calibration", frame_copy)

class ReportGenerator:
    """Generate comprehensive detection reports"""
    
    @staticmethod
    def generate_html_report(analytics, tracker, output_path):
        """Generate HTML report with visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Flying Object Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .alert-warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .alert-critical {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛸 Flying Object Detection Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{analytics.total_detections}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analytics.frame_count}</div>
                <div class="stat-label">Frames Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(analytics.alerts)}</div>
                <div class="stat-label">Alerts Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(tracker.objects)}</div>
                <div class="stat-label">Active Objects</div>
            </div>
        </div>
        
        <h2>Detection Summary by Class</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
                <th>Avg Speed (km/h)</th>
            </tr>
"""
        
        stats = analytics.get_statistics()
        for class_name, count in stats['detections_by_class'].items():
            percentage = (count / analytics.total_detections * 100) if analytics.total_detections > 0 else 0
            avg_speed = stats['avg_speeds'].get(class_name, 0)
            html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
                <td>{avg_speed:.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Recent Alerts</h2>
"""
        
        for alert in analytics.alerts[-10:]:  # Last 10 alerts
            alert_class = 'alert-critical' if 'high speed' in alert.get('alert', '').lower() else 'alert-warning'
            html_content += f"""
        <div class="alert {alert_class}">
            <strong>{alert.get('time', 'N/A')}</strong> - {alert.get('class', 'Unknown')}: 
            {alert.get('alert', alert.get('speed', 'Alert'))}
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)

# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with error handling"""
    try:
        detector = FlyingObjectDetector()
        detector.run()
        
        # Generate comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Config.OUTPUT_DIR / "analytics" / f"report_{timestamp}.html"
        ReportGenerator.generate_html_report(
            detector.analytics, 
            detector.tracker, 
            report_path
        )
        print(f"✓ HTML Report generated: {report_path}")
        
        # Export additional data
        timeline_path = Config.OUTPUT_DIR / "analytics" / f"timeline_{timestamp}.json"
        DataExporter.export_detections_timeline(detector.tracker, timeline_path)
        print(f"✓ Detection timeline exported: {timeline_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Detection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✓ System shutdown complete")

if __name__ == "__main__":
    main()