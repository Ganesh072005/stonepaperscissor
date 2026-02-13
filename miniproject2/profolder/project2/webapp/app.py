import os
import sys
import time
import threading
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

# Ensure yolo_gemini module can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Prevent Matplotlib cache permission errors
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, render_template

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from yolo_gemini.detector import (
    load_yolo_model,
    load_gemini,
    call_gemini,
    draw_boxes,
)
from yolo_gemini.utils import cinfo, cwarn

app = Flask(__name__)

# ================================
# GLOBALS
# ================================
model = None
gemini = None
cap: Optional[cv2.VideoCapture] = None

source_type: str = "none"  # webcam | droidcam | video | none
video_file_path: Optional[str] = None

known_labels: Dict[str, str] = {}     # label → description text
thumbnails: Dict[str, str] = {}       # label → base64 thumbnail
last_gemini_error: Optional[str] = None

fps = 0.0
fps_last_time = time.time()
lock = threading.Lock()

uploads_dir = Path(__file__).resolve().parent / "uploads"
uploads_dir.mkdir(exist_ok=True)


# ================================
# HELPER FUNCTIONS
# ================================
def open_webcam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam index 0 unavailable.")
    return cam


def open_droidcam():
    for idx in (1, 2):
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            return cam
        cam.release()
    raise RuntimeError("DroidCam not found. Start the DroidCam app.")


def open_video(path: str):
    cam = cv2.VideoCapture(path)
    if not cam.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cam


def ensure_capture():
    global cap

    if cap is not None and cap.isOpened():
        return cap

    if source_type == "webcam":
        cap = open_webcam()
    elif source_type == "droidcam":
        cap = open_droidcam()
    elif source_type == "video":
        if not video_file_path:
            raise RuntimeError("Video path not set.")
        cap = open_video(video_file_path)

    return cap


def to_data_url(img):
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        return None
    b64 = base64.b64encode(buffer.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def process_frame(frame):
    """
    Runs YOLO, draws boxes, updates labels, thumbnails,
    and calls Gemini Vision (with the screenshot).
    """
    global known_labels, thumbnails, last_gemini_error, fps

    # YOLO inference (silent mode)
    results = model.predict(frame, verbose=False)

    detections = []
    if results:
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                label = names.get(cls, str(cls))
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy
                })

    # Draw YOLO boxes
    annotated = draw_boxes(frame.copy(), detections)

    # Update thumbnails + Gemini calls
    with lock:
        new_labels = []

        for det in detections:
            label = det["label"]

            # Crop thumbnail safely
            x1, y1, x2, y2 = map(int, det["bbox"])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(1, min(x2, w))
            y2 = max(1, min(y2, h))

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                try:
                    crop = cv2.resize(crop, (160, 160))
                    thumbnails[label] = to_data_url(crop)
                except:
                    pass

            # New label → call Gemini
            if label not in known_labels:
                new_labels.append(label)

        # Run Gemini Vision ONCE for batch of new labels
        if new_labels and gemini:
            try:
                description = call_gemini(gemini, new_labels, frame)
                if not description:
                    description = "(No description returned)"
                for lbl in new_labels:
                    known_labels[lbl] = description
            except Exception as e:
                last_gemini_error = str(e)
                for lbl in new_labels:
                    known_labels[lbl] = "(Gemini error)"
        elif new_labels:
            # Gemini disabled
            for lbl in new_labels:
                known_labels[lbl] = "(Enable GEMINI_API_KEY for descriptions)"

    return annotated


def calculate_fps():
    global fps, fps_last_time
    current = time.time()
    dt = current - fps_last_time
    fps_last_time = current
    if dt > 0:
        fps = 1.0 / dt


def gen_frames():
    """Video streaming generator."""
    while True:
        if source_type == "none":
            # Black screen with instructions
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Select Webcam / DroidCam / Video File",
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            annotated = frame
        else:
            cap = ensure_capture()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            calculate_fps()

            annotated = process_frame(frame)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


# ================================
# ROUTES
# ================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        items = [{
            "label": label,
            "text": text,
            "thumb": thumbnails.get(label)
        } for label, text in known_labels.items()]

    return jsonify({
        "items": items,
        "fps": fps,
        "gemini": bool(gemini),
        "source": source_type,
        "gemini_error": last_gemini_error,
    })


@app.route("/set_source", methods=["POST"])
def set_source():
    global source_type, video_file_path, cap

    data = request.get_json() or {}
    src = data.get("type")
    path = data.get("path")

    # Reset camera
    if cap:
        try: cap.release()
        except: pass
        cap = None

    if src == "webcam":
        source_type = "webcam"
        video_file_path = None
    elif src == "droidcam":
        source_type = "droidcam"
        video_file_path = None
    elif src == "video":
        if not path:
            return jsonify({"ok": False, "error": "Video path required"}), 400
        source_type = "video"
        video_file_path = path
    else:
        return jsonify({"ok": False, "error": "Invalid source type"}), 400

    return jsonify({"ok": True, "source": source_type})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    global video_file_path, source_type, cap

    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    save_path = uploads_dir / f.filename
    f.save(str(save_path))

    # Reset camera
    if cap:
        try: cap.release()
        except: pass
        cap = None

    video_file_path = str(save_path)
    source_type = "video"

    return jsonify({"ok": True, "path": video_file_path})


@app.route("/gemini_enabled")
def gemini_enabled():
    return jsonify({"enabled": bool(gemini)})


# ================================
# INIT MODELS
# ================================
def init_models():
    global model, gemini

    if model is None:
        print(cinfo("Loading YOLO model..."))
        model = load_yolo_model("")

    if gemini is None:
        gemini = load_gemini()
        if not gemini:
            print(cwarn("⚠ Gemini disabled — set GEMINI_API_KEY"))


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    init_models()
    app.run(host="127.0.0.1", port=8000, debug=False)
