import cv2
import mediapipe as mp
import numpy as np
import joblib
from math import acos, degrees
import xgboost as xgb
import pandas as pd

# Load the trained model
model = joblib.load(r'C:\Users\malek\Downloads\rps_xgboost_model.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Feature extraction functions (must match exactly what was used in training)
def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))

def compute_features(landmarks):
    points = np.array([[lm.x, lm.y] for lm in landmarks])

    wrist = points[0]
    tips = [4, 8, 12, 16, 20]  # thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
    mcp = [2, 5, 9, 13, 17]    # metacarpophalangeal joints
    pip = [3, 6, 10, 14, 18]   # proximal interphalangeal joints

    # 1. Distances from wrist to fingertips (5 features)
    dist_wrist_fingertips = [distance(wrist, points[i]) for i in tips]

    # 2. Spread between adjacent fingertips (3 features)
    spreads = [
        distance(points[tips[1]], points[tips[2]]),  # index-middle
        distance(points[tips[2]], points[tips[3]]),  # middle-ring
        distance(points[tips[3]], points[tips[4]])   # ring-pinky
    ]

    # 3. Finger curls (5 features)
    curls = []
    for i in range(5):
        finger_length = distance(points[mcp[i]], wrist)
        if finger_length > 0:  # avoid division by zero
            curl = distance(points[tips[i]], points[mcp[i]]) / finger_length
        else:
            curl = 0
        curls.append(curl)

    # 4. Joint angles (5 features)
    joint_angles = [angle(points[mcp[i]], points[pip[i]], points[tips[i]]) for i in range(5)]

    # 5. Hand shape metrics (2 features)
    hand_width = distance(points[4], points[20])
    hand_length = distance(wrist, points[12])
    aspect_ratio = hand_width / hand_length if hand_length != 0 else 0

    # 6. Binary flags (2 features)
    is_hand_open = int(all(distance(points[i], wrist) > 0.1 for i in tips[1:]))  # fingers extended
    thumb_crossed = int(points[4][0] < points[3][0])  # thumb crosses palm

    # Combine all features in the EXACT SAME ORDER as training
    features = {
        'dist_wrist_thumbtip': dist_wrist_fingertips[0],
        'dist_wrist_indextip': dist_wrist_fingertips[1],
        'dist_wrist_midtip': dist_wrist_fingertips[2],
        'dist_wrist_ringtip': dist_wrist_fingertips[3],
        'dist_wrist_pinkytip': dist_wrist_fingertips[4],
        'spread_index_middle': spreads[0],
        'spread_middle_ring': spreads[1],
        'spread_ring_pinky': spreads[2],
        'curl_thumb': curls[0],
        'curl_index': curls[1],
        'curl_middle': curls[2],
        'curl_ring': curls[3],
        'curl_pinky': curls[4],
        'angle_thumb': joint_angles[0],
        'angle_index': joint_angles[1],
        'angle_middle': joint_angles[2],
        'angle_ring': joint_angles[3],
        'angle_pinky': joint_angles[4],
        'hand_width': hand_width,
        'aspect_ratio': aspect_ratio,
        'is_hand_open': is_hand_open,
        'thumb_crossed': thumb_crossed
    }
    
    return features

# Map numeric predictions to labels (adjust based on how your model was trained)
label_map = {
    0: '🪨 Rock',
    1: '📄 Paper',
    2: '✂️ Scissors'
}

# Open webcam
cap = cv2.VideoCapture(0)
print("🚀 Starting prediction... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
            
            try:
                # Extract features as dictionary
                features_dict = compute_features(hand_landmarks.landmark)
                
                # Convert to DataFrame to ensure correct feature names and order
                features_df = pd.DataFrame([features_dict])
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    # If model has predict_proba (like sklearn wrapper)
                    proba = model.predict_proba(features_df)[0]
                    pred_class = model.predict(features_df)[0]
                    confidence = max(proba)
                else:
                    # For native XGBoost model
                    dmatrix = xgb.DMatrix(features_df)
                    pred_class = model.predict(dmatrix)[0]
                    confidence = 1.0  # XGBoost predictions are already probabilities
                
                # Get label and confidence
                label = label_map.get(int(pred_class), "Unknown")
                conf_text = f"{confidence:.1%}" if 'confidence' in locals() else ""
                
                # Display prediction
                cv2.putText(frame, f"{label} {conf_text}", (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Error in prediction", (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display instructions
    cv2.putText(frame, "Show your hand - Press 'q' to quit", (20, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Rock-Paper-Scissors Live", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()