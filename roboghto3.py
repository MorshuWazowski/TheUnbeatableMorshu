import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
def isolate_hand(frame: np.ndarray, landmarks) -> np.ndarray:
    """Returns cropped hand region with black background.
    Args:
        frame: Input BGR image
        landmarks: MediaPipe hand landmarks
    Returns:
        Cropped BGR image with black padding
    """
    pass
# Suppress TensorFlow/MediaPipe warnings (optional)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load your trained RPS CNN model
model = tf.keras.models.load_model(r'C:\Users\malek\Downloads\rps_be22er_model.h5')  # Replace with your model path
class_names = ['rock', 'paper', 'scissors']  # Update if your classes differ

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Tracking variables
prev_landmarks = None
motion_threshold = 0.1
stable_frames = 0
required_stable_frames = 5
y_threshold = 0.7  # Midpoint for UP/DOWN detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror frame
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks (optional)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Get wrist position
        wrist_y = hand_landmarks.landmark[0].y
        hand_position = "UP" if wrist_y < y_threshold else "DOWN"

        # Motion detection
        if prev_landmarks is not None:
            motion = sum(
                abs(prev.x - curr.x) + abs(prev.y - curr.y)
                for prev, curr in zip(prev_landmarks.landmark, hand_landmarks.landmark)
            )

            stable_frames = stable_frames + 1 if motion < motion_threshold else 0

            # Predict when hand is stable and DOWN
            if stable_frames >= required_stable_frames and hand_position == "DOWN":
                # Extract hand ROI
                x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    hand_roi = isolate_hand(frame, results.multi_hand_landmarks[0])

                # Crop and preprocess for CNN
                hand_roi = frame[y_min:y_max, x_min:x_max]
                if hand_roi.size > 0:  # Check if ROI is valid
                    hand_roi = cv2.resize(hand_roi, (160, 160))  # Match model input size
                    hand_roi = hand_roi / 255.0
                    hand_roi = np.expand_dims(hand_roi, axis=0)

                    # Predict gesture
                    prediction = model.predict(hand_roi, verbose=1)
                    move = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction)

                    # Display prediction
                    cv2.putText(frame, f"Move: {move} ({confidence:.0%})", 
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Predicted: {move} (Confidence: {confidence:.0%})")

                stable_frames = 0  # Reset after prediction

        prev_landmarks = hand_landmarks

        # Display status
        cv2.putText(frame, f"Position: {hand_position}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion: {'STABLE' if stable_frames >= required_stable_frames else 'MOVING'}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('RPS Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
def isolate_hand(frame, landmarks, padding_ratio=0.3):
    # Get landmark extremes
    x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
    y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Dynamic padding (percentage of hand size)
    width = x_max - x_min
    height = y_max - y_min
    padding = int(max(width, height) * padding_ratio)
    
    # Expand ROI but stay within frame
    x1 = max(0, int(x_min) - padding)
    y1 = max(0, int(y_min) - padding)
    x2 = min(frame.shape[1], int(x_max) + padding)
    y2 = min(frame.shape[0], int(y_max) + padding)
    
    # Create black background
    mask = np.zeros_like(frame)
    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]  # Only hand region has data
    
    return mask[y1:y2, x1:x2]  # Return cropped hand with black borders

cap.release()
cv2.destroyAllWindows()