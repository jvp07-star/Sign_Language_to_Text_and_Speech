import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Tasks Engine
TASK_FILE = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Changed to single image mode for browser packets
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

try:
    import processor
except ImportError:
    processor = None

app = Flask(__name__)

MODEL_PATH = "isl_skeleton_model.h5"
LABELS = ["Hello", "Thank You", "Yes", "No", "A", "B", "C"]

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[+] Model loaded successfully!")
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        model = None
else:
    model = None

@app.route('/')
def index():
    return render_template('index.html')

# --- New endpoint that processes video frames directly from Safari ---
@app.route('/process_frame', methods=['POST'])
def process_browser_frame():
    if model is None:
        return jsonify({'english': 'No Model', 'hindi': '...', 'kannada': '...'})
        
    try:
        data = request.json or {}
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({'english': 'Scanning...', 'hindi': '...', 'kannada': '...'})
            
        # Decode base64 image sent by Safari
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks using the modern Tasks API
        result = detector.detect(mp_image)
        en_text = "Scanning..."
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            
            # Build skeleton image
            hand_img = np.zeros((h, w, 3), dtype=np.uint8)
            for start, end in processor.HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(hand_img, points[start], points[end], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(hand_img, pt, 2, (255, 255, 255), -1)
                
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            padding = 20
            x_min, x_max = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
            y_min, y_max = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)
            
            cropped = hand_img[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                resized = cv2.resize(cropped, (300, 300))
                img_array = np.expand_dims(resized, axis=0) / 255.0
                predictions = model.predict(img_array, verbose=0)
                max_index = np.argmax(predictions[0])
                if predictions[0][max_index] > 0.50:
                    en_text = LABELS[max_index]

        # Translate outputs
        hi_text = "..."
        kn_text = "..."
        if en_text != "Scanning...":
            try:
                hi_text = GoogleTranslator(source='en', target='hi').translate(en_text)
                kn_text = GoogleTranslator(source='en', target='kn').translate(en_text)
            except Exception:
                pass
                
        return jsonify({'english': en_text, 'hindi': hi_text, 'kannada': kn_text})
    except Exception as e:
        return jsonify({'english': 'Error', 'hindi': str(e), 'kannada': '...'})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json or {}
    text = data.get('text', '')
    lang = data.get('lang', 'en')
    if not text or text == "...":
        return jsonify({'status': 'empty'})
    try:
        tts = gTTS(text=text, lang=lang)
        os.makedirs("static", exist_ok=True)
        tts.save("static/speech.mp3")
        return jsonify({'status': 'success', 'audio_url': '/static/speech.mp3'})
    except Exception as e:

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

