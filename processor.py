import cv2
import numpy as np

IMG_SIZE = 300
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

def get_hand_skeleton(frame, hand_landmarks):
    h, w, _ = frame.shape
    hand_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Calculate bounding box coordinates from landmarks
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    padding = 20
    x_min, x_max = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
    y_min, y_max = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)

    # Draw lines (Green) and joints (White)
    for start, end in HAND_CONNECTIONS:
        cv2.line(hand_img, points[start], points[end], (0, 255, 0), 2)
    for pt in points:
        cv2.circle(hand_img, pt, 2, (255, 255, 255), -1)

    # Crop and Resize to 300x300
    cropped = hand_img[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))


