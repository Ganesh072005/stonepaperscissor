"""
flying_speed_pro.py
Description:
  • YOLOv8 video detection + multi-object tracking + speed estimation
  • Runs on CPU (no CUDA required)
  • Saves annotated video and CSV logs
  • Supports pixel→meter conversion OR optional homography warping for true planar speed

Requirements:
  pip install ultralytics opencv-python numpy

Tips:
  • Set MODEL_PATH and SOURCE to your paths below.
  • If using webcam, set SOURCE = 0
  • If you know scale (pixels_per_meter) set it. Else speeds shown in px/s.
  • For advanced: fill H_SRC/H_DST to enable homography-based ground-plane speeds.
"""

import os
import time
import csv
import math
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ============================
# CONFIG
# ============================

# ---- Paths ----
MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\weights\generalized_40_class\last.pt"  # your trained weights
SOURCE = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\bird.mp4"  # or 0 for webcam


# Output directory (auto-created)
OUT_DIR = r"C:\Users\harsh\OneDrive\Desktop\miniproject\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Detection params ----
CONFIDENCE = 0.35
IOU_NMS    = 0.5
IMG_SIZE   = 640
DEVICE     = 'cpu'  # force CPU

# ---- Tracking params ----
MAX_LOST_FRAMES = 30      # how many frames to keep "lost" tracks
ASSOC_IOU_THRESH = 0.3    # detection↔track association threshold
SMOOTH_ALPHA     = 0.7    # 0..1, higher = smoother (EMA over centers & speed)
HISTORY_LEN      = 20     # how many past centers to keep for trail & smoothing

# ---- Speed conversion (simple scale) ----
# If you know that X pixels ~= 1 meter in your scene (approx or measured),
# set PIXELS_PER_METER and we'll compute km/h. Otherwise stays px/s.
PIXELS_PER_METER = 40.0   # tune for your scene; set None to disable km/h
SHOW_KMPH        = True   # overlay km/h if PIXELS_PER_METER is set

# ---- Homography (advanced; optional)
# If you know 4 source image points on the ground and their corresponding
# 4 target real-plane points (e.g., meters), we can warp centers to a
# metric plane and compute true speeds without global scale.
# Leave H_SRC/H_DST = None to disable homography mode.
H_SRC = None
H_DST = None
# Example (uncomment & edit):
# H_SRC = np.float32([[100,700],[1200,700],[100,900],[1200,900]]) # image pts
# H_DST = np.float32([[0,0],[30,0],[0,10],[30,10]])               # meters

# ---- Rendering/HUD ----
DRAW_TRAILS  = True
TRAIL_COLOR  = (255, 220, 60)
BOX_COLOR    = (0, 220, 0)
TEXT_COLOR   = (50, 255, 255)
ID_COLOR     = (255, 120, 120)
THICKNESS    = 2
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# ============================
# UTILS
# ============================

def iou_xyxy(a, b):
    """ IoU between two boxes [x1,y1,x2,y2]. """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1 + 1)) * max(0, (ay2 - ay1 + 1))
    area_b = max(0, (bx2 - bx1 + 1)) * max(0, (by2 - by1 + 1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def exponential_smooth(prev_val, new_val, alpha=0.7):
    """
    Smooths either scalars or 2D tuples (x, y) using exponential moving average.
    """
    if prev_val is None:
        return new_val

    # If the values are tuples (like (x, y)), smooth each component separately
    if isinstance(prev_val, (tuple, list)) and isinstance(new_val, (tuple, list)):
        return (
            alpha * prev_val[0] + (1 - alpha) * new_val[0],
            alpha * prev_val[1] + (1 - alpha) * new_val[1],
        )
    else:
        # Fallback for numeric smoothing
        return (alpha * prev_val) + ((1 - alpha) * new_val)


def compute_homography(src_pts, dst_pts):
    """Compute homography H (image->metric)."""
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    return H

def warp_point(H, p):
    """Apply homography H to point p = (x,y)."""
    xy1 = np.array([[p[0], p[1], 1.0]], dtype=np.float32).T
    w = H @ xy1
    if w[2, 0] != 0:
        return (w[0, 0] / w[2, 0], w[1, 0] / w[2, 0])
    else:
        return (p[0], p[1])

# ============================
# TRACKER
# ============================

class Track:
    __slots__ = ("track_id", "cls_id", "label", "box", "center",
                 "center_smooth", "speed_px_s", "speed_px_s_smooth",
                 "lost_frames", "history")

    def __init__(self, track_id, cls_id, label, box, center):
        self.track_id = track_id
        self.cls_id   = cls_id
        self.label    = label
        self.box      = box  # [x1,y1,x2,y2]
        self.center   = center
        self.center_smooth = None
        self.speed_px_s = 0.0
        self.speed_px_s_smooth = 0.0
        self.lost_frames = 0
        self.history = deque(maxlen=HISTORY_LEN)

class SimpleTracker:
    """Greedy IoU association with exponential smoothing & velocity estimate."""
    def __init__(self, iou_thresh=0.3, max_lost=30, smooth_alpha=0.7):
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self.alpha      = smooth_alpha
        self.tracks     = {}
        self.next_id    = 1
        self.last_timestamp = None

    def _center_of(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, det_boxes, det_cls, det_labels, now_ts, homography=None, fps=None):
        """
        det_boxes: list of [x1,y1,x2,y2]
        det_cls:   list of int
        det_labels:list of str
        now_ts:    current timestamp (time.time())
        homography: optional H to warp centers to metric plane
        fps: frames-per-second (for delta-t fallback)
        """
        # Compute dt
        if self.last_timestamp is None:
            dt = 0.0
        else:
            dt = max(1e-6, now_ts - self.last_timestamp)
        self.last_timestamp = now_ts

        # Mark all current tracks as "unmatched"
        for t in self.tracks.values():
            t.lost_frames += 1

        # Association (greedy by IoU)
        unmatched_dets = set(range(len(det_boxes)))
        for tid, t in list(self.tracks.items()):
            # find best detection for this track
            best_iou, best_j = 0.0, -1
            for j in unmatched_dets:
                iou = iou_xyxy(t.box, det_boxes[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_j >= 0 and best_iou >= self.iou_thresh:
                # match
                box = det_boxes[best_j]
                center = self._center_of(box)
                # speed in pixel/sec (or warped-plane units/sec if homography provided)
                if dt > 0:
                    prev_c = t.center_smooth if t.center_smooth is not None else t.center
                    if homography is not None:
                        # warp both points to metric plane
                        pc = warp_point(homography, prev_c)
                        cc = warp_point(homography, center)
                        dist = math.hypot(cc[0] - pc[0], cc[1] - pc[1])
                    else:
                        dist = math.hypot(center[0] - prev_c[0], center[1] - prev_c[1])
                    speed = dist / dt
                else:
                    speed = 0.0

                # updates
                t.box = box
                t.center = center
                t.center_smooth = (
                    exponential_smooth(t.center_smooth, center, self.alpha)
                )
                t.speed_px_s = speed
                t.speed_px_s_smooth = exponential_smooth(t.speed_px_s_smooth, speed, self.alpha)
                t.lost_frames = 0
                t.cls_id = det_cls[best_j]
                t.label  = det_labels[best_j]
                t.history.append(t.center_smooth if t.center_smooth is not None else t.center)

                unmatched_dets.remove(best_j)

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            box = det_boxes[j]
            center = self._center_of(box)
            tr = Track(self.next_id, det_cls[j], det_labels[j], box, center)
            tr.center_smooth = center
            tr.history.append(center)
            self.tracks[self.next_id] = tr
            self.next_id += 1

        # Remove stale tracks
        to_delete = [tid for tid, t in self.tracks.items() if t.lost_frames > self.max_lost]
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks

# ============================
# MAIN
# ============================

def main():
    # Validate paths
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not (SOURCE == 0 or os.path.exists(SOURCE)):
        raise FileNotFoundError(f"Source not found: {SOURCE}")

    print("\nLoading model...")
    model = YOLO(MODEL_PATH)

    # Prepare capture
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise IOError(f"Could not open source: {SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        # fallback for webcams or files without fps metadata
        fps = 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Prepare output writer
    base_name = "webcam" if SOURCE == 0 else os.path.splitext(os.path.basename(SOURCE))[0]
    out_vid_path = os.path.join(OUT_DIR, f"{base_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_vid_path, fourcc, fps, (width, height))

    # CSV logger
    csv_path = os.path.join(OUT_DIR, f"{base_name}_speeds.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "track_id", "label", "x", "y", "speed_px_s", "speed_km_h"])

    # Homography (optional)
    H = None
    if H_SRC is not None and H_DST is not None:
        H = compute_homography(np.float32(H_SRC), np.float32(H_DST))
        print("Homography enabled (image → metric plane).")

    tracker = SimpleTracker(iou_thresh=ASSOC_IOU_THRESH,
                            max_lost=MAX_LOST_FRAMES,
                            smooth_alpha=SMOOTH_ALPHA)

    frame_idx = 0
    t0 = time.time()
    fps_smooth = None

    print(f"Source opened at {width}x{height} @ {fps:.1f} FPS. Processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        tic = time.time()

        # Inference
        results = model.predict(
            source=frame, imgsz=IMG_SIZE, conf=CONFIDENCE, iou=IOU_NMS,
            device=DEVICE, verbose=False
        )

        det_boxes, det_cls, det_labels = [], [], []
        # Gather detections
        for r in results:
            if r.boxes is None: 
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                det_boxes.append([x1, y1, x2, y2])
                c = int(b.cls)
                det_cls.append(c)
                det_labels.append(model.names[c])

        # Update tracker
        tracks = tracker.update(det_boxes, det_cls, det_labels, time.time(), homography=H, fps=fps)

        # Draw
        for tid, t in tracks.items():
            x1, y1, x2, y2 = map(int, t.box)
            cx, cy = (t.center_smooth if t.center_smooth is not None else t.center)
            cx_i, cy_i = int(cx), int(cy)

            # Convert speed to km/h (either via homography units or pixels→meters)
            speed_kmh = None
            if H is not None:
                # In homography mode, t.speed_px_s_smooth is actually "metric units per second"
                # (because we warp centers to metric plane before distance).
                speed_mps = t.speed_px_s_smooth
                speed_kmh = speed_mps * 3.6
            elif PIXELS_PER_METER and PIXELS_PER_METER > 0:
                speed_mps = t.speed_px_s_smooth / PIXELS_PER_METER
                speed_kmh = speed_mps * 3.6

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

            # Trail
            if DRAW_TRAILS and len(t.history) >= 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in t.history], dtype=np.int32)
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], TRAIL_COLOR, 2)

            # ID + Label + Speed text
            id_text = f"ID {tid}"
            label_text = f"{t.label}"
            if speed_kmh is not None and SHOW_KMPH:
                speed_text = f"{speed_kmh:.1f} km/h"
            else:
                speed_text = f"{t.speed_px_s_smooth:.1f} px/s"

            text = f"{id_text} | {label_text} | {speed_text}"
            cv2.putText(frame, text, (x1, max(20, y1 - 8)), FONT, 0.55, TEXT_COLOR, 2)

            # CSV log (use current center)
            csv_w.writerow([frame_idx, tid, t.label, f"{cx:.2f}", f"{cy:.2f}",
                            f"{t.speed_px_s_smooth:.4f}", f"{speed_kmh:.4f}" if speed_kmh is not None else ""])

        # HUD: FPS, counts
        toc = time.time()
        inst_fps = 1.0 / max(1e-6, (toc - tic))
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (15, 28), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Objects: {len(tracks)}", (15, 58), FONT, 0.7, (255, 255, 255), 2)

        # Write & show
        writer.write(frame)
        cv2.imshow("Flying Objects - YOLOv8 Speed Pro (CPU)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Stopped by user.")
            break

    # Cleanup
    cap.release()
    writer.release()
    csv_f.close()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print("\nDone!")
    print(f" Process time: {elapsed:.2f}s, frames: {frame_idx}, avg FPS: {frame_idx/max(1e-6,elapsed):.1f}")
    print(f"Saved video: {out_vid_path}")
    print(f"CSV log:     {csv_path}")

if __name__ == "__main__":
    main()
