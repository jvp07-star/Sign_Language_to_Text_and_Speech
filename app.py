import atexit
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf
from deep_translator import GoogleTranslator
from flask import Flask, render_template, Response, jsonify
from gtts import gTTS
from spellchecker import SpellChecker

from processor import get_hand_skeleton

app = Flask(__name__)

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_skeleton_model.h5")
TASK_ASSET_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONFIDENCE_THRESHOLD = 0.80
HOLD_DURATION = 1.5
SPACE_DURATION = 2.5

# --- Core Engines Initialization ---
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
spell = SpellChecker()

# Engine for English offline speech
engine = pyttsx3.init()
engine.setProperty('rate', 145)
engine_lock = threading.Lock()

# --- Global Tracking States ---
predicted_letter = "Searching..."
current_word = ""
final_sentence = []

# Translation Buffers (Only update when sentence pieces change)
hindi_translation = ""
kannada_translation = ""

# Time Keeping variables for Debouncer
tracking_letter = ""
letter_start_time = None
hand_absent_start_time = None
cooldown_until = 0.0


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("CRITICAL: Camera failed to open!")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.video.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror mode
                with self.lock:
                    self.frame = frame.copy()
            time.sleep(0.02)

    def release(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()


cam = VideoCamera()


def speak_indic_worker(text, lang_code):
    """Safely plays audio on a background worker thread."""
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        temp_file = f"temp_{lang_code}.mp3"
        tts.save(temp_file)
        if os.name == 'nt':
            os.system(f'start /min "" "{temp_file}"')
        else:
            os.system(f'afplay "{temp_file}" &' if os.uname().sysname == 'Darwin' else f'mpg123 "{temp_file}" &')
    except Exception as e:
        print(f"Voice Synthesis Error: {e}")


def run_tts_english(text):
    with engine_lock:
        engine.say(text)
        engine.runAndWait()


def trigger_sentence_translation():
    """Triggered dynamically only when words are committed into full sentence arrays."""
    global final_sentence, hindi_translation, kannada_translation
    sentence_str = " ".join(final_sentence)
    if sentence_str.strip() == "":
        hindi_translation = ""
        kannada_translation = ""
        return

    try:
        hindi_translation = GoogleTranslator(source='en', target='hi').translate(sentence_str)
        kannada_translation = GoogleTranslator(source='en', target='kn').translate(sentence_str)
    except Exception as e:
        print(f"Translation Failure: {e}")


def gen_frames():
    global predicted_letter, current_word, final_sentence
    global tracking_letter, letter_start_time, hand_absent_start_time, cooldown_until

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK_ASSET_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame = cam.frame
            if frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect_for_video(mp_image, int(current_time * 1000))

            if results.hand_landmarks and model:
                hand_absent_start_time = None
                raw_landmarks = results.hand_landmarks[0]

                # Render skeleton structures and return 300x300 bounding box segment
                skeleton = get_hand_skeleton(frame, raw_landmarks)

                # Classify gesture frame
                pred = model.predict(np.expand_dims(skeleton, axis=0), verbose=0)
                prob_distribution = pred[0]
                max_idx = np.argmax(prob_distribution)

                if prob_distribution[max_idx] > CONFIDENCE_THRESHOLD:
                    predicted_letter = LABELS[max_idx]
                else:
                    predicted_letter = "Uncertain"

                # Word Assembly State Debouncer
                if predicted_letter not in ["Uncertain", "No Hand Detected"] and current_time > cooldown_until:
                    if predicted_letter == tracking_letter:
                        elapsed = current_time - letter_start_time
                        progress_w = int((elapsed / HOLD_DURATION) * 200)

                        # UI Feedback Bar for user visual processing lock
                        cv2.rectangle(frame, (50, 130), (50 + min(progress_w, 200), 145), (0, 255, 0), -1)

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
                        cv2.putText(frame, "Switching...", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                predicted_letter = "No Hand Detected"
                tracking_letter = ""
                letter_start_time = None

                # Process sentence space delay calculation
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

                        # Trigger background translation automatically after sentence changes
                        threading.Thread(target=trigger_sentence_translation, daemon=True).start()

            # Render Heads-Up-Display straight onto stream
            cv2.putText(frame, f"Live: {predicted_letter}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Word: {current_word}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    return jsonify({
        'live': predicted_letter,
        'word': current_word,
        'en': " ".join(final_sentence),
        'hi': hindi_translation,
        'kn': kannada_translation
    })

# Action endpoints for word modifications remain here
@app.route('/action/delete_word', methods=['POST'])
def delete_word():
    global final_sentence
    if final_sentence:
        final_sentence.pop()
        threading.Thread(target=trigger_sentence_translation, daemon=True).start()
    return jsonify({'status': 'success'})

@app.route('/action/clear_all', methods=['POST'])
def clear_all():
    global final_sentence, current_word, hindi_translation, kannada_translation
    final_sentence = []
    current_word = ""
    hindi_translation = ""
    kannada_translation = ""
    return jsonify({'status': 'success'})

@atexit.register
def release_camera_on_exit():
    if 'cam' in globals():
        cam.release()
    cv2.destroyAllWindows()
    print("Webcam hardware and window contexts successfully released.")


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=7860)