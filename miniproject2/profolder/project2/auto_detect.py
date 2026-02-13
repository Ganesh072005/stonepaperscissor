import os
from pathlib import Path
from typing import Optional, Set

import cv2

from yolo_gemini.detector import load_yolo_model, load_gemini, call_gemini, draw_boxes
from yolo_gemini.utils import cinfo, cwarn, cerror, csuccess


def pick_source() -> Optional[cv2.VideoCapture]:
    print("\nSELECT INPUT SOURCE:")
    print("1 → Video File")
    print("2 → DroidCam (index 1 or 2)")
    print("3 → Laptop Webcam (index 0)\n")
    choice = input("Enter choice [1/2/3]: ").strip()

    cap: Optional[cv2.VideoCapture] = None
    if choice == "1":
        path = input("Enter full video file path: ").strip().strip('"')
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(cerror(f"Failed to open video: {path}"))
            return None
        return cap
    elif choice == "2":
        for idx in (1, 2):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(csuccess(f"Opened DroidCam at index {idx}"))
                return cap
            if cap:  # ensure release
                cap.release()
        print(cerror("Could not open DroidCam (tried indices 1 and 2)."))
        return None
    elif choice == "3":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(cerror("Failed to open laptop webcam at index 0."))
            return None
        return cap
    else:
        print(cwarn("Invalid choice."))
        return None


def main() -> int:
    try:
        model = load_yolo_model("")
    except Exception as e:
        print(cerror(str(e)))
        return 1

    gemini = load_gemini()
    described: Set[str] = set()

    cap = pick_source()
    if cap is None:
        return 2

    win_title = "YOLOv8 + Gemini Real-Time Detection"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(cwarn("Stream ended or failed to read frame."))
                break

            results = model(frame)
            if not results:
                cv2.imshow(win_title, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            r = results[0]
            detections = []
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

            # Draw on frame
            annotated = draw_boxes(frame.copy(), detections)

            # Call Gemini once per NEW label
            for det in detections:
                label = det["label"]
                if label not in described and gemini is not None:
                    text = call_gemini(gemini, [label])
                    if text:
                        print(cinfo(f"Gemini on '{label}': {text}"))
                    described.add(label)

            cv2.imshow(win_title, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(csuccess("Real-time detection finished."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
