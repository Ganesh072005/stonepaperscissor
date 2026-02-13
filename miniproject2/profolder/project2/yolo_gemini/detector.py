import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import (
    cinfo,
    csuccess,
    cwarn,
    cerror,
    ensure_output_dir,
    is_image_file,
    list_images,
    timestamp,
    safe_output_path,
)

# Gemini SDK
try:
    import google.generativeai as genai
except Exception:
    genai = None


# ================================
# 1. Default YOLO weights location
# ================================
def _default_weights_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "weights" / "best.pt"



# ================================
# 2. Load YOLOv8 model
# ================================
def load_yolo_model(weights_path: str = "") -> YOLO:
    wp = Path(weights_path) if weights_path else _default_weights_path()
    if not wp.exists():
        raise FileNotFoundError(
            f"Weights not found at '{wp}'. Place best.pt or pass --weights."
        )
    print(cinfo(f"Loading YOLOv8 model from {wp}..."))
    model = YOLO(str(wp))
    print(csuccess("YOLOv8 model loaded."))
    return model



# ================================
# 3. Load Gemini model (2.5 Flash)
# ================================
def load_gemini(model_name: str = "gemini-2.5-flash"):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(cwarn("GEMINI_API_KEY not set → Gemini disabled"))
        return None

    if genai is None:
        print(cwarn("google-generativeai missing → Gemini disabled"))
        return None

    genai.configure(api_key=api_key)

    candidates = [
        model_name,
        "gemini-2.5-flash",
        "gemini-flash-latest",
    ]

    last_err = None
    for name in candidates:
        try:
            mdl = genai.GenerativeModel(name)
            print(csuccess(f"Gemini model initialized: {name}"))
            return mdl
        except Exception as e:
            last_err = e
            continue

    print(cwarn(f"Gemini model failed: {last_err}"))
    return None



# ================================
# 4. Convert frame → PNG bytes
# ================================
def frame_to_png_bytes(frame):
    """Convert BGR OpenCV frame → PNG bytes for Gemini Vision."""
    ok, buff = cv2.imencode(".png", frame)
    if not ok:
        return None
    return buff.tobytes()



# ================================
# 5. YOLO + Gemini combined reasoning
# ================================
def call_gemini(gemini_model, labels: List[str], frame=None) -> Optional[str]:
    """
    Sends BOTH:
      • YOLO labels
      • Full screenshot frame
    to Gemini for an intelligent multi-sentence description.
    """
    if not gemini_model:
        if labels:
            return f"Detected: {', '.join(labels)}."
        return None

    prompt_text = (
        "You are an advanced vision-language model.\n"
        "You will receive an image and a list of on-screen objects detected by YOLO.\n\n"
        "Your tasks:\n"
        "• Describe the entire scene clearly.\n"
        "• Explain what the detected objects are doing.\n"
        "• Infer context, environment, or actions.\n"
        "• Mention safety risks or unusual behavior.\n"
        "• Provide 3–6 sentences of detailed reasoning.\n\n"
        f"YOLO detected: {', '.join(labels)}\n"
        "Now analyze the image below:"
    )

    # Input structure for Gemini: [text, image_bytes]
    parts = [prompt_text]

    if frame is not None:
        png_bytes = frame_to_png_bytes(frame)
        if png_bytes:
            parts.append({
                "mime_type": "image/png",
                "data": png_bytes
            })

    try:
        resp = gemini_model.generate_content(parts)

        # Gemini v2.5 returns .text
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()

        # Fallback if structured output
        if hasattr(resp, "candidates"):
            cand = resp.candidates[0]
            if cand and cand.content.parts:
                return cand.content.parts[0].text.strip()

        return None

    except Exception as e:
        print(cwarn(f"Gemini Vision failed: {e}"))
        if labels:
            return f"Detected: {', '.join(labels)}."
        return None



# ================================
# 6. Draw YOLO bounding boxes
# ================================
def draw_boxes(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['label']} {det['confidence']:.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x1, y1 - th - bl), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(image_bgr, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image_bgr



# ================================
# 7. YOLO inference helper
# ================================
def _yolo_infer(model: YOLO, image_path: str):
    results = model(image_path)
    if not results:
        return None
    return results[0]



# ================================
# 8. Convert YOLO output → dict
# ================================
def _extract_detections(result) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    if getattr(result, "boxes", None) is None:
        return detections

    names = result.names if hasattr(result, "names") else {}

    for box in result.boxes:
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        xyxy = box.xyxy[0].tolist()
        label = names.get(cls_id, str(cls_id))

        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
        })

    return detections



# ================================
# 9. Single image detection API
# ================================
def detect_single_image(model: YOLO, image_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = ensure_output_dir(options.get("output_dir", "outputs"))
    gemini = options.get("gemini")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(cinfo(f"Running YOLO on: {image_path}"))

    result = _yolo_infer(model, image_path)
    detections = _extract_detections(result) if result else []

    img = cv2.imread(image_path)
    annotated = draw_boxes(img.copy(), detections)

    stem = Path(image_path).stem
    annotated_path = safe_output_path(out_dir, f"annotated_{stem}", ".jpg")
    cv2.imwrite(annotated_path, annotated)

    print(csuccess(f"Saved annotated → {annotated_path}"))

    # Labels + full frame → Gemini
    labels = sorted({d["label"] for d in detections})
    summary = call_gemini(gemini, labels, img)

    records = [{
        "image": Path(image_path).name,
        "label": d["label"],
        "confidence": d["confidence"],
        "bbox": d["bbox"]
    } for d in detections]

    return {
        "image": image_path,
        "annotated": annotated_path,
        "summary": summary,
        "records": records,
    }



# ================================
# 10. Folder batch detection
# ================================
def detect_folder(model: YOLO, folder_path: str, options: Dict[str, Any]):
    out_dir = ensure_output_dir(options.get("output_dir", "outputs"))
    gemini = options.get("gemini")
    save_json = options.get("save_json", False)
    save_csv = options.get("save_csv", False)

    images = list_images(folder_path)
    if not images:
        print(cwarn("No images found"))
        return {"images": [], "records": []}

    all_records = []
    all_labels = []

    for img_path in images:
        res = detect_single_image(model, img_path, {
            "output_dir": out_dir,
            "gemini": gemini,
        })
        all_records.extend(res["records"])
        all_labels.extend(r["label"] for r in res["records"])

    # Batch summary
    summary = call_gemini(gemini, sorted(set(all_labels)), None)
    if summary:
        print(cinfo(f"Batch Gemini summary:\n{summary}"))

    # Export JSON
    if save_json:
        js = safe_output_path(out_dir, f"detections_{timestamp()}", ".json")
        with open(js, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)
        print(csuccess(f"Saved JSON → {js}"))

    # Export CSV
    if save_csv:
        csv_path = safe_output_path(out_dir, f"detections_{timestamp()}", ".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "confidence", "x1", "y1", "x2", "y2"])
            for r in all_records:
                bbox = r["bbox"]
                writer.writerow([r["image"], r["label"], r["confidence"], *bbox])
        print(csuccess(f"Saved CSV → {csv_path}"))

    return {
        "images": images,
        "records": all_records,
        "summary": summary,
    }
