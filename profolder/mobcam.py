"""
Advanced, optimized Flying Object Detection (YOLOv8)
- Supports: Video file, DroidCam (webcam index), IP Webcam (URL)
- Threaded capture + threaded detection (latest-frame policy)
- MJPG forcing for webcam to avoid green-screen
- Resize + frame-skip + FPS overlay
- Save annotated output optional
"""

import cv2
import time
import os
import threading
from ultralytics import YOLO

# ---------------------------
# CONFIGURATION (edit here)
# ---------------------------

MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\weights\generalized_40_class\best.pt"
VIDEO_FILE_PATH = r"C:\Users\harsh\OneDrive\Desktop\miniproject\profolder\bird.mp4"

# INPUT MODE:
# 1 = Video file
# 2 = DroidCam / system webcam (by index or auto detect)
# 3 = IP Webcam URL (e.g. "http://192.168.0.105:4747/video")
INPUT_MODE = 2

# When INPUT_MODE == 2 and CAMERA_INDEX is None -> auto-detect.
CAMERA_INDEX = None   # set to an int (0/1/2...) to force; or None to auto-scan

IP_WEBCAM_URL = "http://192.168.0.105:4747/video"  # used when INPUT_MODE == 3

# Performance tuning
DEVICE = 'cpu'            # 'cpu' or 'cuda' (if you have GPU)
CONFIDENCE = 0.35
RESIZE_WH = (640, 360)    # width, height for detection (smaller => faster)
PROCESS_EVERY_N_FRAMES = 1  # 1 = process every frame (detection thread works on latest frame anyway)
VIDEO_SAVE = True
OUTPUT_PATH = "output_annotated.mp4"  # saved annotated video (if VIDEO_SAVE True)

# Misc
FORCE_MJPG = True  # force MJPG when using webcam to avoid green-screen issues
MAX_CAMERA_INDEX_SCAN = 8

# ---------------------------
# Helper: threaded video capture
# ---------------------------

class VideoCaptureThread:
    """ Threaded VideoCapture that always keeps the latest frame. """
    def __init__(self, source, force_mjpg=False, width=None, height=None):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        # try force mjpg for webcams
        if force_mjpg:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # end of stream or camera disconnected
                self.running = False
                break
            with self.lock:
                self.latest_frame = frame
        # release handled by close()

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            # return a copy to avoid race
            return self.latest_frame.copy()

    def is_running(self):
        return self.running

    def close(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass

# ---------------------------
# Helper: auto-detect working camera index
# ---------------------------

def find_working_camera(max_index=8, force_mjpg=False):
    """Scan camera indices and return first index that returns a valid frame."""
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        if force_mjpg:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return i
    return None

# ---------------------------
# Main detection manager
# ---------------------------

def main():
    # 1) Check model path
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    # 2) Load model (this may take time)
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)  # loads to CPU or CUDA depending on device usage below
    print("Model loaded.")

    # 3) Prepare input source
    if INPUT_MODE == 1:
        print("Using video file as input.")
        source = VIDEO_FILE_PATH
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        capture = VideoCaptureThread(source, force_mjpg=False)  # file input, no mjpg forced

    elif INPUT_MODE == 2:
        print("Using DroidCam / system webcam as input.")
        cam_index = CAMERA_INDEX
        if cam_index is None:
            print("Auto-detecting webcam index...")
            cam_index = find_working_camera(max_index=MAX_CAMERA_INDEX_SCAN, force_mjpg=FORCE_MJPG)
            if cam_index is None:
                raise IOError("No working webcam found. Please connect DroidCam or specify CAMERA_INDEX.")
            print(f"Auto-detected webcam index: {cam_index}")
        else:
            print(f"Using user-specified camera index: {cam_index}")

        capture = VideoCaptureThread(cam_index, force_mjpg=FORCE_MJPG,
                                    width=RESIZE_WH[0], height=RESIZE_WH[1])

    elif INPUT_MODE == 3:
        print("Using IP Webcam URL as input.")
        source = IP_WEBCAM_URL
        capture = VideoCaptureThread(source, force_mjpg=False)

    else:
        raise ValueError("Invalid INPUT_MODE. Choose 1=video,2=webcam,3=ip")

    # 4) Prepare output writer if requested
    writer = None
    out_fps = 20.0  # default output fps
    time.sleep(0.3)  # short warm-up for capture thread
    sample_frame = capture.read_latest()
    if sample_frame is not None:
        h, w = sample_frame.shape[:2]
        out_fps = max(10.0, capture.cap.get(cv2.CAP_PROP_FPS) or 20.0)
    else:
        # fallback dims
        w, h = RESIZE_WH

    if VIDEO_SAVE:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, out_fps, (w, h))
        if not writer.isOpened():
            print("Warning: Could not open output writer, disabling save.")
            writer = None
        else:
            print(f"Saving annotated output to: {OUTPUT_PATH} (fps={out_fps:.1f})")

    # 5) Detector thread logic (works on latest frame)
    annotated_frame_lock = threading.Lock()
    annotated_frame = None
    stop_event = threading.Event()

    # For speed tracking (center positions stored by unique instance id)
    previous_positions = {}
    object_speeds = {}

    def detector_loop():
        nonlocal annotated_frame, previous_positions, object_speeds
        frame_idx = 0
        while not stop_event.is_set() and capture.is_running():
            frame = capture.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_idx += 1
            if PROCESS_EVERY_N_FRAMES > 1 and (frame_idx % PROCESS_EVERY_N_FRAMES) != 0:
                # skip processing but keep latest annotated_frame unchanged
                time.sleep(0.001)
                continue

            # Resize for detection speed
            small = cv2.resize(frame, RESIZE_WH)

            # Run detection (use model.predict or model(small) — using predict for consistency)
            try:
                results = model.predict(source=small, conf=CONFIDENCE, device=DEVICE, verbose=False)
            except Exception as e:
                print("Detection error:", e)
                time.sleep(0.05)
                continue

            # Annotate detections on the small frame, then upscale to original capture size for display
            display = small.copy()
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    label = model.names[cls] if hasattr(model, 'names') else str(cls)
                    # box.xyxy is a tensor-like; convert to python floats
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # speed: pixels/sec on the resized frame scale
                    fps_est = max(1.0, capture.cap.get(cv2.CAP_PROP_FPS) or 20.0)
                    frame_time = 1.0 / fps_est
                    if label in previous_positions:
                        dx = cx - previous_positions[label][0]
                        dy = cy - previous_positions[label][1]
                        dist = (dx*dx + dy*dy) ** 0.5
                        object_speeds[label] = dist / frame_time
                    else:
                        object_speeds[label] = 0.0
                    previous_positions[label] = (cx, cy)

                    # draw
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"{label}: {object_speeds[label]:.1f}px/s", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # Upscale annotated small frame back to capture frame size for consistent display / save
            # If capture frame equals small size, skip resize
            if frame.shape[1] != display.shape[1] or frame.shape[0] != display.shape[0]:
                annotated = cv2.resize(display, (frame.shape[1], frame.shape[0]))
            else:
                annotated = display

            with annotated_frame_lock:
                annotated_frame = annotated

            # small sleep to let UI thread run
            time.sleep(0.001)

    # start detector thread
    detector_thread = threading.Thread(target=detector_loop, daemon=True)
    detector_thread.start()

    # 6) Main display loop (reads annotated_frame and shows it)
    win_name = "Flying Object Detection (YOLOv8) - Press Q to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps_smooth = 0.0
    display_count = 0

    try:
        while capture.is_running():
            frame_to_show = None
            with annotated_frame_lock:
                if annotated_frame is not None:
                    frame_to_show = annotated_frame.copy()

            if frame_to_show is None:
                # no annotated frame yet; show raw latest frame (scaled)
                raw = capture.read_latest()
                if raw is None:
                    time.sleep(0.01)
                    continue
                frame_to_show = cv2.resize(raw, (w, h))

            # overlay FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

            cv2.putText(frame_to_show, f"FPS: {fps_smooth:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win_name, frame_to_show)
            display_count += 1

            # Save if writer present
            if writer is not None:
                # ensure same size as writer expects
                out_frame = cv2.resize(frame_to_show, (w, h))
                writer.write(out_frame)

            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        print("Stopping...")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    # cleanup
    stop_event.set()
    detector_thread.join(timeout=1.0)
    capture.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Done. Exiting.")

if __name__ == "__main__":
    main()
