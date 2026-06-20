import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import string
import time
import os
from spellchecker import SpellChecker
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# --- Configuration ---
img_size = 300
model_path = "isl_skeleton_model.h5"
CONFIDENCE_THRESHOLD = 0.80
HOLD_DURATION = 1.5
SPACE_DURATION = 2.5

# --- Font Configuration ---
# 'Nirmala.ttf' supports English, Hindi, and Kannada characters natively.
# Update this path if your operating system uses a different path or font.
FONT_PATH = "C:\\Windows\\Fonts\\Nirmala.ttc"
FONT_SIZE = 32

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print(f"Warning: Font file not found at {FONT_PATH}. Falling back to default system font.")
    font = ImageFont.load_default()

# Initialize pyttsx3 for offline English audio engine execution
engine = pyttsx3.init()
engine.setProperty('rate', 145)
spell = SpellChecker()

# Load Model & Labels
model = tf.keras.models.load_model(model_path)
labels = list(string.ascii_uppercase)


def speak_indic(text, lang_code):
    """Generates and plays clear audio output for Hindi/Kannada translations."""
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        temp_file = "temp_translation.mp3"
        tts.save(temp_file)

        if os.name == 'nt':  # Windows
            os.system(f'start /min "" "{temp_file}"')
        else:  # Mac / Linux
            os.system(f'afplay "{temp_file}" &' if os.uname().sysname == 'Darwin' else f'mpg123 "{temp_file}" &')

    except Exception as e:
        print(f"Voice Synthesis Error: {e}")


# ==========================================
# MEDIAPIPE TASKS API INITIALIZATION
# ==========================================
model_asset_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_asset_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,  # Strictly tracking a single hand crop for your model
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Standard 21-point hand joint configurations for direct OpenCV rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)  # Pinky & Palm
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# State Variables
current_word = ""
final_sentence = []
tracking_letter = ""
letter_start_time = None
hand_absent_start_time = None
cooldown_until = 0

hindi_translation = ""
kannada_translation = ""

print("\n--- Translating Sentence Builder Active ---")
print("Shortcuts are mapped directly on the video viewport dashboard window.")

with HandLandmarker.create_from_options(options) as recognizer:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        # Convert frame format for the MediaPipe engine
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = recognizer.detect_for_video(mp_image, timestamp_ms)
        predicted_letter = "No Hand Detected"

        if results.hand_landmarks:
            hand_absent_start_time = None

            for hand_landmarks in results.hand_landmarks:
                x_max, y_max = 0, 0
                x_min, y_min = w, h
                points = []

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                    x_max, x_min = max(x, x_max), min(x, x_min)
                    y_max, y_min = max(y, y_max), min(y, y_min)

                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                # Isolate the hand structure onto a pure black canvas
                hand_img = np.zeros((h, w, 3), dtype=np.uint8)

                # Render joint lines
                for connection in HAND_CONNECTIONS:
                    start_pt = points[connection[0]]
                    end_pt = points[connection[1]]
                    cv2.line(hand_img, start_pt, end_pt, (0, 255, 0), 2)
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

                # Render joint tracking nodes
                for pt in points:
                    cv2.circle(hand_img, pt, 3, (255, 255, 255), -1)
                    cv2.circle(frame, pt, 3, (0, 0, 255), -1)

                # Crop skeleton slice safely for Custom Keras Classifier
                try:
                    cropped = hand_img[y_min:y_max, x_min:x_max]
                    canvas = cv2.resize(cropped, (img_size, img_size))
                except Exception:
                    canvas = cv2.resize(hand_img, (img_size, img_size))

                test_img = np.expand_dims(canvas, axis=0) / 255.0
                predictions = model.predict(test_img, verbose=0)
                prob_distribution = predictions[0]
                max_index = np.argmax(prob_distribution)

                if prob_distribution[max_index] > CONFIDENCE_THRESHOLD:
                    predicted_letter = labels[max_index]
                else:
                    predicted_letter = "Uncertain"

                # Word Assembly Debounce Logic
                if predicted_letter not in ["Uncertain", "No Hand Detected"] and current_time > cooldown_until:
                    if predicted_letter == tracking_letter:
                        elapsed = current_time - letter_start_time
                        progress_w = int((elapsed / HOLD_DURATION) * 200)
                        cv2.rectangle(frame, (20, 110), (20 + min(progress_w, 200), 120), (0, 255, 0), -1)

                        if elapsed >= HOLD_DURATION:
                            current_word += predicted_letter
                            cooldown_until = current_time + 1.2
                            tracking_letter = ""
                            letter_start_time = None
                    else:
                        tracking_letter = predicted_letter
                        letter_start_time = current_time
                else:
                    if current_time <= cooldown_until:
                        cv2.putText(frame, "Switch Gesture...", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255),
                                    2)
        else:
            tracking_letter = ""
            letter_start_time = None

            if current_word != "":
                if hand_absent_start_time is None:
                    hand_absent_start_time = current_time
                elif current_time - hand_absent_start_time >= SPACE_DURATION:
                    raw_word = current_word.lower()
                    corrected_word = spell.correction(raw_word) or raw_word
                    corrected_word = corrected_word.upper()

                    final_sentence.append(corrected_word)
                    current_word = ""
                    hand_absent_start_time = None

        display_sentence = " ".join(final_sentence)

        # HUD Overlays (Standard English Labels via OpenCV)
        cv2.putText(frame, f"Live: {predicted_letter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Building Word: {current_word}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Full HD Bottom Panel Height (Positioned perfectly at h - 220)
        cv2.rectangle(frame, (0, h - 220), (w, h), (0, 0, 0), -1)
        cv2.line(frame, (w - 420, h - 220), (w - 420, h), (100, 100, 100), 1)

        # Convert back to PIL for clean native Unicode rendering
        cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Left Column Panel: Full HD Spacing for Translations (65 pixels per row)
        draw.text((35, h - 200), f"EN: {display_sentence}", font=font, fill=(255, 255, 255))
        draw.text((35, h - 135), f"HI: {hindi_translation}", font=font, fill=(0, 255, 0))
        draw.text((35, h - 70), f"KN: {kannada_translation}", font=font, fill=(255, 255, 0))

        # Right Column Panel: Full HD Shortcut Map Layout (Shifted right to fit w - 420 boundaries)
        draw.text((w - 390, h - 210), "[Enter] Translate", font=font, fill=(200, 200, 200))
        draw.text((w - 390, h - 178), "[E] Speak English", font=font, fill=(255, 255, 255))
        draw.text((w - 390, h - 146), "[H] Speak Hindi", font=font, fill=(0, 255, 0))
        draw.text((w - 390, h - 114), "[K] Speak Kannada", font=font, fill=(255, 255, 0))
        draw.text((w - 390, h - 82), "[Backspace] Del Word", font=font, fill=(255, 100, 100))
        draw.text((w - 390, h - 50), "[Delete] Clear All", font=font, fill=(255, 50, 50))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Sign Language Sentence Engine", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

        elif key == 13 or key == 10:  # Enter Key: Generates Hindi & Kannada text conversions
            if display_sentence.strip() != "":
                try:
                    hindi_translation = GoogleTranslator(source='en', target='hi').translate(display_sentence)
                    kannada_translation = GoogleTranslator(source='en', target='kn').translate(display_sentence)
                except Exception as e:
                    print(f"Translation Error: {e}")
                    hindi_translation = "Translation Error"
                    kannada_translation = "Translation Error"

        elif key == 8:  # Backspace Key: Drops the last added word matrix element
            if final_sentence:removed_word = final_sentence.pop()
            print(f"Removed last word: {removed_word}")
            hindi_translation = ""
            kannada_translation = ""

        elif key == 255 or key == 46:  # Delete Key: Flushes out text cache buffers completely
            final_sentence = []
            current_word = ""
            hindi_translation = ""
            kannada_translation = ""
            print("Reset all active workspace string sequences.")

        # Audio Engine Playback & Auto-Purge Listeners
        elif key in [ord('e'), ord('E')]:  # Speak English
            if display_sentence.strip() != "":
                engine.say(display_sentence)
                engine.runAndWait()

        elif key in [ord('h'), ord('H')]:  # Speak Hindi
            if hindi_translation.strip() and "Error" not in hindi_translation:
                speak_indic(hindi_translation, 'hi')

        elif key in [ord('k'), ord('K')]:  # Speak Kannada
            if kannada_translation.strip() and "Error" not in kannada_translation:
                speak_indic(kannada_translation, 'kn')

cap.release()
cv2.destroyAllWindows()

# Safety disk garbage cleanup tracking
if os.path.exists("temp_translation.mp3"):
    try: os.remove("temp_translation.mp3")
    except: pass