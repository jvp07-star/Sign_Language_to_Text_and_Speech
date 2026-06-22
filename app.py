import os, cv2, threading, time, numpy as np, tensorflow as tf, mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from processor import get_hand_skeleton
from deep_translator import GoogleTranslator

app = Flask(__name__)

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_skeleton_model.h5")
TASK_ASSET_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Load AI model
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
current_detected_text = "Searching..."

class VideoCamera:
    def __init__(self):
        # Using AVFOUNDATION for macOS compatibility
        self.video = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.video.isOpened():
            print("CRITICAL: Camera failed to open! Check macOS Privacy Settings.")
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            ret, frame = self.video.read()
            if ret:
                with self.lock: self.frame = frame.copy()
            time.sleep(0.02)

cam = VideoCamera()

def gen_frames():
    global current_detected_text
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK_ASSET_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1
    )
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame = cam.frame
            if frame is None: continue
            
            # 1. Detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            # 2. Processing
            if results.hand_landmarks and model:
                landmarks = results.hand_landmarks[0]
                skeleton = get_hand_skeleton(landmarks)
                # Prediction shape (1, 63)
                pred = model.predict(np.expand_dims(skeleton, axis=0), verbose=0)
                current_detected_text = LABELS[np.argmax(pred[0])]
                
                # Feedback on stream
                cv2.putText(frame, current_detected_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            # 3. Stream
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    try:
        hi = GoogleTranslator(source='en', target='hi').translate(current_detected_text)
        kn = GoogleTranslator(source='en', target='kn').translate(current_detected_text)
        return jsonify({'en': current_detected_text, 'hi': hi, 'kn': kn})
    except: return jsonify({'en': current_detected_text, 'hi': "...", 'kn': "..."})

if __name__ == '__main__':
    print("Starting Server... Check your terminal for Camera Errors.")
    app.run(debug=False, threaded=True, port=5001)

    