# yolo_gemini_detector

YOLOv8 + Gemini AI Detection Suite for Windows. Detect objects in images/folders and in real-time (video/DroidCam/webcam). Export JSON/CSV, get Gemini AI descriptions, and optional TTS narration.

## Features
- Image/folder detection with YOLOv8
- Real-time video/DroidCam/webcam detection
- Annotated outputs saved under `outputs/`
- JSON/CSV export of bounding boxes
- Gemini AI descriptions for detected labels
- Optional text-to-speech (pyttsx3)
- Colored terminal logs and simple timers
- Windows-friendly paths and examples

## Project Structure
```
project/
│── auto_detect.py          # Real-time video/DroidCam/webcam + Gemini
│── detect_ai.py            # Image/folder batch + Gemini
│── setup.py
│── README.md
│── requirements.txt
│── .gitignore
│
├── weights/
│     └── best.pt           # place your weights file here
│
├── outputs/
│     (auto-saved detection results)
│
└── yolo_gemini/
      ├── __init__.py
      ├── cli.py
      ├── detector.py
      ├── utils.py
```

## Installation
1) Create and activate a virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies
```powershell
pip install -r project\requirements.txt
```

3) Place your YOLOv8 weights at `project\weights\best.pt` (or pass `--weights` to select another file).

4) Set your Gemini API key (required for Gemini features)
- Copy the example environment file:
```powershell
copy project\.env.example project\.env
```
- Open `project\.env` and replace `your_gemini_api_key_here` with your actual API key from Google AI Studio.
> Get an API key: https://ai.google.dev/

5) (Optional) Install package locally for console script
```powershell
pip install -e project
```

## Usage
### Batch: Top-level script
```powershell
# Single image
python project\detect_ai.py --image path\to\img.jpg --save-json --save-csv --speak

# Folder of images
python project\detect_ai.py --folder path\to\images --save-json --save-csv
```

### Batch: CLI after install (console script `yoloai`)
```powershell
yoloai --image path\to\image.jpg --save-json --speak
```

### Real-time: auto_detect.py
```powershell
python project\auto_detect.py
```
Then choose the input source:
- 1 → Video File (enter full path)
- 2 → DroidCam (tries indices 1 then 2)
- 3 → Laptop Webcam (index 0)

Suggested Windows alias for quick start:
```powershell
doskey yoloauto=python C:\full\path\to\auto_detect.py $*
```
Run it simply as:
```powershell
yoloauto
```

### Common arguments (batch mode)
- `--image`: Path to a single image
- `--folder`: Path to a folder of images
- `--weights`: Path to YOLOv8 weights (default: `project\weights\best.pt`)
- `--output-dir`: Output directory (default: `project\outputs`)
- `--save-json`: Save detections as JSON
- `--save-csv`: Save detections as CSV
- `--speak`: Narrate Gemini summary with TTS

## Example output files
- Annotated image: `outputs/annotated_<image_stem>.jpg`
- JSON: `outputs/detections_<timestamp>.json`
- CSV: `outputs/detections_<timestamp>.csv`

## Notes
- If `GEMINI_API_KEY` is not set, Gemini features are skipped gracefully.
- If `pyttsx3` or audio drivers are missing, TTS is skipped.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
