import argparse
from pathlib import Path

from .detector import (
    load_yolo_model,
    load_gemini,
    detect_single_image,
    detect_folder,
)
from .utils import cinfo, cerror


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLOv8 + Gemini image/folder detector",
    )
    p.add_argument("--image", type=str, default=None, help="Path to a single image")
    p.add_argument("--folder", type=str, default=None, help="Path to a folder of images")
    p.add_argument("--weights", type=str, default=None, help="Path to YOLOv8 weights (default: project/weights/best.pt)")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs (default: project/outputs)")
    p.add_argument("--save-json", action="store_true", help="Export detections to JSON")
    p.add_argument("--save-csv", action="store_true", help="Export detections to CSV")
    p.add_argument("--speak", action="store_true", help="Speak Gemini summary using TTS")
    p.add_argument("--no-gemini", action="store_true", help="Disable Gemini calls")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.image and not args.folder:
        print(cerror("You must provide either --image or --folder"))
        return 2

    try:
        model = load_yolo_model(args.weights or "")
        gemini = None if args.no_gemini else load_gemini()
        options = {
            "output_dir": args.output_dir,
            "save_json": args.save_json,
            "save_csv": args.save_csv,
            "speak": args.speak,
            "gemini": gemini,
        }

        if args.image:
            print(cinfo(f"Processing single image: {args.image}"))
            detect_single_image(model, args.image, options)
        else:
            print(cinfo(f"Processing folder: {args.folder}"))
            detect_folder(model, args.folder, options)
        return 0
    except Exception as e:
        print(cerror(f"Error: {e}"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
