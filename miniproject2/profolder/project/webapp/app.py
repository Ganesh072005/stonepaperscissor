import os
import sys
from pathlib import Path
import threading
import time
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# Ensure the project directory is on sys.path so `yolo_gemini` can be imported when running from webapp/
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Configure a writable Matplotlib config dir early (Ultralytics may import matplotlib under the hood)
try:
    mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
except Exception:
    pass

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from yolo_gemini.detector import load_yolo_model, load_gemini, call_gemini, draw_boxes, detect_single_image
from yolo_gemini.utils import cinfo, cwarn

app = Flask(__name__)

# Globals for stream state
cap: Optional[cv2.VideoCapture] = None
model = None
gemini = None

# State shared with /status
known_labels: Dict[str, str] = {}
thumbs: Dict[str, str] = {}  # label -> data URL (base64 JPG)
lock = threading.Lock()

# Source state
source_type: str = "none"  # none | webcam | droidcam | video
video_source_path: Optional[str] = None
uploads_dir = Path(__file__).resolve().parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

# Gemini error state
last_gemini_error: Optional[str] = None


def _open_webcam() -> cv2.VideoCapture:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Unable to open laptop webcam at index 0. Close other apps using the camera.")
    return cam


def _open_droidcam() -> cv2.VideoCapture:
    for idx in (1, 2):
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            return cam
        cam.release()
    raise RuntimeError("Unable to open DroidCam (tried indices 1 and 2). Start DroidCam and try again.")


def _open_video(path: str) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(path)
    if not cam.isOpened():
        raise RuntimeError(f"Unable to open video file: {path}")
    return cam


def ensure_capture() -> cv2.VideoCapture:
    global cap
    if cap is not None and cap.isOpened():
        return cap
    if source_type == "webcam":
        cap = _open_webcam()
    elif source_type == "droidcam":
        cap = _open_droidcam()
    elif source_type == "video":
        if not video_source_path:
            raise RuntimeError("Video source path not set. Upload a file first.")
        cap = _open_video(video_source_path)
    else:
        cap = _open_webcam()
    return cap


def to_data_url(image_bgr) -> str:
    _, buf = cv2.imencode('.jpg', image_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


def to_data_url_file(path: str) -> str:
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


def process_frame(frame):
    global known_labels, thumbs
    results = model(frame)
    detections: List[Dict] = []
    if results:
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        if getattr(r, "boxes", None) is not None and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else [0, 0, 0, 0]
                label = names.get(cls_id, str(cls_id))
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                })

    annotated = draw_boxes(frame.copy(), detections)

    # Update Gemini descriptions and thumbnails
    with lock:
        for det in detections:
            label = det["label"]
            # Thumbnail crop
            x1, y1, x2, y2 = map(int, det["bbox"])
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (160, 160))
                    thumbs[label] = to_data_url(crop)

        # Call Gemini once per new label
        new_labels = [l for l in {d['label'] for d in detections} if l not in known_labels]
        if new_labels and gemini is not None:
            try:
                text = call_gemini(gemini, new_labels)
            except Exception as e:
                text = None
                globals()['last_gemini_error'] = str(e)
            if text:
                for l in new_labels:
                    known_labels[l] = text
            else:
                for l in new_labels:
                    known_labels[l] = "(Gemini description unavailable)"
        elif new_labels:
            # Gemini disabled; still record placeholders
            for l in new_labels:
                known_labels[l] = "(Set GEMINI_API_KEY to enable descriptions)"

    return annotated


def gen_frames():
    while True:
        if source_type == "none":
            # Render placeholder frame asking user to select source
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = "Select source: Webcam / DroidCam / Upload Video"
            cv2.putText(frame, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            annotated = frame
        else:
            cam = ensure_capture()
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
            annotated = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/select')
def select():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with lock:
        items = []
        for label, text in known_labels.items():
            items.append({
                "label": label,
                "text": text,
                "thumb": thumbs.get(label)
            })
    return jsonify({
        "items": items,
        "gemini": bool(gemini is not None),
        "source": source_type,
        "gemini_error": last_gemini_error,
    })


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return render_template('analyze.html', result=None)
    # POST: handle image upload and run detection
    file = request.files.get('file')
    if not file:
        return render_template('analyze.html', result={
            'error': 'Please upload an image file.'
        })
    # Save uploaded file
    save_path = uploads_dir / file.filename
    file.save(str(save_path))
    # Ensure models ready
    init_models()
    # Run detection and summary
    try:
        res = detect_single_image(model, str(save_path), {
            'output_dir': str(uploads_dir),
            'speak': False,
            'gemini': gemini,
        })
        annotated_path = res.get('annotated')
        desc = res.get('summary')
        records = res.get('records', [])
        annotated_data_url = to_data_url_file(annotated_path) if annotated_path else None
        return render_template('analyze.html', result={
            'annotated_data_url': annotated_data_url,
            'summary': desc,
            'records': records,
            'filename': file.filename,
        })
    except Exception as e:
        return render_template('analyze.html', result={
            'error': f'Analysis failed: {e}'
        })


@app.route('/gemini_enabled')
def gemini_enabled():
    return jsonify({"enabled": bool(gemini is not None)})


@app.route('/set_source', methods=['POST'])
def set_source():
    global source_type, video_source_path, cap
    data = request.get_json(silent=True) or {}
    src = str(data.get('type', 'webcam')).lower()
    path = data.get('path')
    # Release existing capture if any
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None
    # Switch
    if src == 'webcam':
        source_type = 'webcam'
        video_source_path = None
    elif src == 'droidcam':
        source_type = 'droidcam'
        video_source_path = None
    elif src == 'video':
        if not path:
            return jsonify({"ok": False, "error": "path required for video"}), 400
        source_type = 'video'
        video_source_path = path
    else:
        return jsonify({"ok": False, "error": "invalid source type"}), 400
    return jsonify({"ok": True, "source": source_type})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global source_type, video_source_path, cap
    f = request.files.get('file')
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    # Save to uploads dir
    save_path = uploads_dir / f.filename
    f.save(str(save_path))
    # Switch source to this file
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None
    source_type = 'video'
    video_source_path = str(save_path)
    return jsonify({"ok": True, "path": video_source_path})


def init_models():
    global model, gemini
    if model is None:
        print(cinfo("Loading YOLOv8 model for webapp..."))
        model = load_yolo_model("")
    if gemini is None:
        gemini = load_gemini()
        if gemini is None:
            print(cwarn("Gemini disabled or not configured."))


if __name__ == '__main__':
    init_models()
    # Default to no source; user will choose in UI
    app.run(host='127.0.0.1', port=8000, debug=False)
