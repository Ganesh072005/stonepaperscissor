"""
extract_frames.py
-----------------
This script performs two tasks using OpenCV:
1. Extracts frames from a saved video file.
2. Captures and saves frames from a live camera feed.

Author: Harsha M G
Project: Real-Time Flying Object Detection (YOLOv8)
"""

import cv2
import os
import time

# ==========================
# 1️⃣  EXTRACT FRAMES FROM A VIDEO FILE
# ==========================

def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save extracted frames.
        frame_interval (int): Save one frame every 'frame_interval' frames.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {saved_count} frames from '{video_path}' into '{output_folder}'.")


# ==========================
# 2️⃣  CAPTURE FRAMES FROM LIVE CAMERA INPUT
# ==========================

def capture_frames_from_camera(output_folder, save_interval=2):
    """
    Captures frames from webcam and saves them periodically.

    Args:
        output_folder (str): Directory to save captured frames.
        save_interval (int): Time (seconds) between frame saves.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("🎥 Press 'q' to stop capturing.")
    last_save_time = time.time()
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame. Exiting...")
            break

        # Display the live feed
        cv2.imshow("Live Camera Feed", frame)

        # Save frame every 'save_interval' seconds
        if time.time() - last_save_time >= save_interval:
            filename = os.path.join(output_folder, f"camera_frame_{frame_index:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"💾 Saved {filename}")
            frame_index += 1
            last_save_time = time.time()

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Capture stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================
# 3️⃣  MAIN PROGRAM
# ==========================

if __name__ == "__main__":
    print("Select Mode:")
    print("1. Extract frames from video file")
    print("2. Capture frames from live camera")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        video_path = input("Enter path to video file: ")
        output_folder = "output_frames_video"
        frame_interval = int(input("Enter frame interval (e.g., 30 for ~1 sec): ") or 1)
        extract_frames_from_video(video_path, output_folder, frame_interval)

    elif choice == "2":
        output_folder = "output_frames_camera"
        save_interval = int(input("Enter save interval in seconds (default 2): ") or 0.01)
        capture_frames_from_camera(output_folder, save_interval)

    else:
        print("❌ Invalid choice. Please enter 1 or 2.")
