import cv2
import numpy as np
import mediapipe as mp

IMG_SIZE = 300
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

def process_frame(frame, detector, frame_count, model, labels):
    h, w, _ = frame.shape
    
    # Tasks API requires an explicit MediaPipe Image object format wrapper
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Calculate synthetic millisecond timestamp increment for video input tracking
    timestamp_ms = int((frame_count * 1000) / 30)
    
    # Run landmark evaluation check
    result = detector.detect_for_video(mp_image, timestamp_ms)
    
    predicted_text = "Scanning..."
    
    if result.hand_landmarks:
        # Grab first found hand data structure matrix
        landmarks = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        # Draw track indicator elements onto webcam feed window
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
            
        # Compile black/green skeleton data
        hand_img = np.zeros((h, w, 3), dtype=np.uint8)
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(hand_img, points[start], points[end], (0, 255, 0), 2)
        for pt in points:
            cv2.circle(hand_img, pt, 2, (255, 255, 255), -1)
            
        # Normalize bounds sizing padding limits
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        padding = 20
        x_min, x_max = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
        y_min, y_max = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)
        
        cropped = hand_img[y_min:y_max, x_min:x_max]
        
        if cropped.size > 0 and model is not None:
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(resized, axis=0) / 255.0
            
            predictions = model.predict(img_array, verbose=0)
            max_index = np.argmax(predictions[0])
            if predictions[0][max_index] > 0.50:
                predicted_text = labels[max_index]
                
    return frame, predicted_text

