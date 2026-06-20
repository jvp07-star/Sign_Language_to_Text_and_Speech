import cv2
import os
import string

# Create base dataset directory
base_dir = "./isl_dataset"
os.makedirs(base_dir, exist_ok=True)

# Generate a list of letters from A to Z
letters = list(string.ascii_uppercase)
letter_index = 0
target_count = 500

cap = cv2.VideoCapture(0)
count = 0
recording = False

print("--- Sign Language Dataset Collector ---")
print("Press 's' to start capturing a letter.")
print("Press 'ESC' at any time to quit.")

while letter_index < len(letters):
    current_letter = letters[letter_index]
    save_dir = os.path.join(base_dir, current_letter)
    os.makedirs(save_dir, exist_ok=True)

    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # UI Overlay text logic
    if recording:
        cv2.putText(display_frame, f"Capturing '{current_letter}': {count}/{target_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame and increment
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        count += 1

        # Check if burst is finished
        if count >= target_count:
            print(f" Saved {target_count} images for letter {current_letter}!")
            recording = False
            count = 0
            letter_index += 1  # Move to next letter automatically
    else:
        cv2.putText(display_frame, f"Next up: Letter '{current_letter}'", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, "Form sign & press 'S' to start", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Multi-Letter Data Collector", display_frame)

    key = cv2.waitKey(1)
    if key == ord('s') and not recording:
        recording = True
    elif key == 27:  # ESC key to stop the entire program
        break

print("\n Dataset collection complete! Check your './isl_dataset' folder.")
cap.release()
cv2.destroyAllWindows()
