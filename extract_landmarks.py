import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
input_base_dir = "./isl_dataset"
output_base_dir = "./processed_dataset"
img_size = 300  # Standardized dimensions for the neural network
MODEL_PATH = 'hand_landmarker.task'

# --- Automatic Model Downloader ---
if not os.path.exists(MODEL_PATH):
    print(f"'{MODEL_PATH}' not found. Downloading from Google's servers...")
    # Updated official URL using the 'latest' endpoint to prevent HTTP 404 errors
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

    # Custom User-Agent header added to prevent firewalls or security software from blocking Python's download request
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )

    try:
        with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading the model file: {e}")
        print("\nPlease download it manually via your web browser from:")
        print(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"Then place the saved file directly into your directory: {os.getcwd()}")
        exit()

# --- Define Hand Connections for Skeleton Drawing ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (9, 10), (10, 11), (11, 12),  # Middle
    (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm base
]

# Initialize Modern MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# Verify source directory exists
if not os.path.exists(input_base_dir):
    print(f"Error: Source directory '{input_base_dir}' not found. Please run the collector script first.")
    exit()

print("Starting landmark extraction processing...")

# Loop through each letter folder (A-Z)
for letter_folder in sorted(os.listdir(input_base_dir)):
    input_letter_path = os.path.join(input_base_dir, letter_folder)

    # Skip files, only process subdirectories
    if not os.path.isdir(input_letter_path):
        continue

    output_letter_path = os.path.join(output_base_dir, letter_folder)
    os.makedirs(output_letter_path, exist_ok=True)

    print(f"Processing Letter: {letter_folder}...")

    for img_name in os.listdir(input_letter_path):
        img_path = os.path.join(input_letter_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert image to MediaPipe standard format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        # Default fallback canvas (blank black image)
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Check if any hands were detected
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Find bounding box coordinates to crop/center the hand dynamically
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y

                # Add padding around the hand skeleton
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # Draw the connections manually onto a scratch image
                hand_img = np.zeros((h, w, 3), dtype=np.uint8)

                # 1. Draw connection lines (Green)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection

                    start_lm = hand_landmarks[start_idx]
                    end_lm = hand_landmarks[end_idx]

                    pt1 = (int(start_lm.x * w), int(start_lm.y * h))
                    pt2 = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(hand_img, pt1, pt2, (0, 255, 0), 2)

                # 2. Draw joints/landmarks (White)
                for lm in hand_landmarks:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(hand_img, pt, 2, (255, 255, 255), -1)

                # Crop to the hand area and resize to uniform target dimensions (300x300)
                try:
                    cropped_hand = hand_img[y_min:y_max, x_min:x_max]
                    # Ensure cropped section isn't empty before resizing
                    if cropped_hand.size > 0:
                        canvas = cv2.resize(cropped_hand, (img_size, img_size))
                    else:
                        canvas = cv2.resize(hand_img, (img_size, img_size))
                except Exception:
                    canvas = cv2.resize(hand_img, (img_size, img_size))

            # Save the processed landmark image
            output_img_path = os.path.join(output_letter_path, img_name)
            cv2.imwrite(output_img_path, canvas)

# Clean up detector instance safely
detector.close()
print("\nLandmark extraction complete! Clean skeletons saved inside './processed_dataset'.")
