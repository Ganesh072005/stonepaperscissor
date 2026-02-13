from setuptools import setup, find_packages

setup(
    name="yolo_gemini_detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python",
        "google-generativeai",
        "pyttsx3",
        "colorama",
        "numpy",
        "pillow",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "yoloai = yolo_gemini.cli:main",
        ]
    }
)
