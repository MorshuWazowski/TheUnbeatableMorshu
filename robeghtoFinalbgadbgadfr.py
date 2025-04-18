import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def normalize_landmarks(landmarks):
    # Convert landmarks to a flat list
    coords = [(lm.x, lm.y) for lm in landmarks]

    # Get wrist and middle finger tip
    x_wrist, y_wrist = coords[0]
    x_tip, y_tip = coords[9]

    scale = np.sqrt((x_tip - x_wrist)**2 + (y_tip - y_wrist)**2)
    if scale == 0:
        return None

    normalized = []
    for x, y in coords:
        normalized.append((x - x_wrist) / scale)
        normalized.append((y - y_wrist) / scale)

    return np.array(normalized).reshape(1, -1)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural interaction and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            pinky_tip = hand_landmarks.landmark[20]
            wrist = hand_landmarks.landmark[0]
            pinkyKnuck = hand_landmarks.landmark[17]
            index_dist = distance(thumb_tip, index_tip)
            middle_dist = distance(thumb_tip, middle_tip)
            wrist_pinky_dist = distance(wrist, pinky_tip)
            pinky_pinkyKnuck_dist = distance(pinkyKnuck, pinky_tip)

            normalized_input = normalize_landmarks(hand_landmarks.landmark)
            if normalized_input is not None:
                prediction = model.predict(normalized_input)[0]
                probabilities = model.predict_proba(normalized_input)[0]
                confidence = np.max(probabilities)
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                
                rock_conf = probabilities[label_encoder.transform(["rock"])[0]]
                paper_conf = probabilities[label_encoder.transform(["paper"])[0]]
                scissors_conf = probabilities[label_encoder.transform(["scissors"])[0]]
                
                if predicted_label == "rock" and (index_dist > 0.2 or middle_dist > 0.2):
                    predicted_label = "paper"

                if predicted_label == "paper" and (wrist_pinky_dist < 0.25 or pinky_pinkyKnuck_dist < 0.15):
                    predicted_label = "scissors"
                
                # Draw prediction on the screen
                text = f"{predicted_label} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
