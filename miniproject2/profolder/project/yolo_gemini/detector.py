import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from dotenv import load_dotenv, find_dotenv
    # Load .env from current or any parent directory for reliability when running from subfolders
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

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore


def _default_weights_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "weights" / "best.pt"


def load_yolo_model(weights_path: str) -> YOLO:
    wp = Path(weights_path) if weights_path else _default_weights_path()
    if not wp.exists():
        raise FileNotFoundError(
            f"Weights not found at '{wp}'. Place your YOLOv8 weights (e.g., best.pt) or pass --weights."
        )
    print(cinfo(f"Loading YOLOv8 model from {wp}..."))
    model = YOLO(str(wp))
    print(csuccess("YOLOv8 model loaded."))
    return model


def load_gemini(model_name: str = "gemini-2.5-flash"):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(cwarn("GEMINI_API_KEY not set. Gemini features will be skipped."))
        return None
    if genai is None:
        print(cwarn("google-generativeai not installed. Gemini features will be skipped."))
        return None
    genai.configure(api_key=api_key)
    candidates = [model_name, "gemini-2.5-flash", "gemini-2.5-flash-8b"]
    last_err = None
    for name in candidates:
        try:
            mdl = genai.GenerativeModel(name)
            print(csuccess(f"Gemini model initialized: {name}"))
            return mdl
        except Exception as e:
            last_err = e
            continue
    print(cwarn(f"Failed to initialize Gemini with any known model names: {candidates}. Last error: {last_err}"))
    return None


def call_gemini(gemini_model, labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    # Local fallback description builder
    def _fallback(ls: List[str]) -> str:
        uniq = ", ".join(sorted(set(ls)))
        return f"Detected objects: {uniq}."

    if not gemini_model:
        return _fallback(labels)
    prompt = (
        "You are assisting with object detection results. "
        "Given a short list of object names, provide a concise, human-friendly summary of what is present.\n\n"
        f"Objects: {', '.join(labels)}\n\n"
        "Respond in 1-2 sentences, focusing on flying objects or relevant items if applicable."
    )
    try:
        result = gemini_model.generate_content(prompt)
        text = getattr(result, "text", None) or (result.candidates[0].content.parts[0].text if getattr(result, "candidates", None) else None)
        return text or _fallback(labels)
    except Exception as e:
        print(cwarn(f"Gemini generation failed: {e}"))
        return _fallback(labels)


def speak_text(text: str) -> None:
    if not text:
        return
    if pyttsx3 is None:
        print(cwarn("pyttsx3 not available; skipping TTS."))
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(cwarn(f"TTS failed: {e}"))


def draw_boxes(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])  # [x1,y1,x2,y2]
        label = f"{det['label']} {det['confidence']:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(image_bgr, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image_bgr


def _yolo_infer(model: YOLO, image_path: str):
    results = model(image_path)
    if not results:
        return None
    return results[0]


def _extract_detections(result) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    names = result.names if hasattr(result, "names") else {}
    if getattr(result, "boxes", None) is None or result.boxes is None:
        return detections
    for box in result.boxes:
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else [0, 0, 0, 0]
        label = names.get(cls_id, str(cls_id))
        detections.append(
            {
                "label": label,
                "confidence": conf,
                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            }
        )
    return detections


def _default_output_dir() -> Path:
    return (Path(__file__).resolve().parents[1] / "outputs").resolve()


def _resolve_output_dir(option_dir: Optional[str]) -> Path:
    if option_dir:
        return Path(ensure_output_dir(option_dir))
    return Path(ensure_output_dir(str(_default_output_dir())))


def export_json(records: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(csuccess(f"Saved JSON: {output_path}"))


def export_csv(records: List[Dict[str, Any]], output_path: str) -> None:
    if not records:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "confidence", "x1", "y1", "x2", "y2"])
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "confidence", "x1", "y1", "x2", "y2"])
            for r in records:
                bbox = r.get('bbox', [0, 0, 0, 0])
                writer.writerow([
                    r.get("image", ""),
                    r.get("label", ""),
                    f"{r.get('confidence', 0.0):.4f}",
                    f"{bbox[0]:.2f}", f"{bbox[1]:.2f}", f"{bbox[2]:.2f}", f"{bbox[3]:.2f}",
                ])
    print(csuccess(f"Saved CSV: {output_path}"))


def detect_single_image(model: YOLO, image_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = _resolve_output_dir(options.get("output_dir"))
    speak = bool(options.get("speak", False))
    gemini = options.get("gemini")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(cinfo(f"Running YOLO on: {image_path}"))
    result = _yolo_infer(model, image_path)
    detections = _extract_detections(result) if result is not None else []

    img = cv2.imread(image_path)
    annotated = draw_boxes(img.copy(), detections)
    stem = Path(image_path).stem
    annotated_path = safe_output_path(out_dir, f"annotated_{stem}", ".jpg")
    cv2.imwrite(annotated_path, annotated)
    print(csuccess(f"Saved annotated: {annotated_path}"))

    labels = sorted({d["label"] for d in detections})
    summary = call_gemini(gemini, labels) if gemini else None
    if summary:
        print(cinfo(f"Gemini summary: {summary}"))
        if speak:
            speak_text(summary)

    image_name = Path(image_path).name
    records: List[Dict[str, Any]] = []
    for d in detections:
        records.append({
            "image": image_name,
            "label": d["label"],
            "confidence": float(d["confidence"]),
            "bbox": [float(x) for x in d["bbox"]],
        })

    return {
        "image": image_path,
        "annotated": annotated_path,
        "summary": summary,
        "records": records,
    }


def detect_folder(model: YOLO, folder_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = _resolve_output_dir(options.get("output_dir"))
    save_json = bool(options.get("save_json", False))
    save_csv = bool(options.get("save_csv", False))
    speak = bool(options.get("speak", False))
    gemini = options.get("gemini")

    images = list_images(folder_path)
    if not images:
        print(cwarn("No images found in folder."))
        return {"images": [], "records": []}

    all_records: List[Dict[str, Any]] = []
    all_labels: List[str] = []

    for img_path in images:
        res = detect_single_image(model, img_path, {"output_dir": str(out_dir), "speak": False, "gemini": gemini})
        all_records.extend(res.get("records", []))
        all_labels.extend([r["label"] for r in res.get("records", [])])

    labels = sorted(set(all_labels))
    summary = call_gemini(gemini, labels) if gemini else None
    if summary:
        print(cinfo(f"Batch Gemini summary: {summary}"))
        if speak:
            speak_text(summary)

    if save_json:
        json_path = safe_output_path(out_dir, f"detections_{timestamp()}", ".json")
        export_json(all_records, json_path)
    if save_csv:
        csv_path = safe_output_path(out_dir, f"detections_{timestamp()}", ".csv")
        export_csv(all_records, csv_path)

    return {
        "images": images,
        "records": all_records,
        "summary": summary,
    }
