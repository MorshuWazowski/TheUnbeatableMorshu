import cv2
import mediapipe as mp
import time
import numpy as np

# Performance-optimized MediaPipe config
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Better for continuous video
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # Lightest model
)

# Memory management
last_frame_time = time.time()
frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)

def classify_gesture(landmarks):
    # Thresholds (adjust based on testing)
    extension_threshold = 0.05
    spread_threshold = 0.1
    
    # Thumb check (more reliable)
    thumb_open = landmarks[4].y < landmarks[2].y - extension_threshold
    
    # Finger extension checks
    fingers = [
        landmarks[8].y < landmarks[6].y - extension_threshold,  # Index
        landmarks[12].y < landmarks[10].y - extension_threshold,  # Middle
        landmarks[16].y < landmarks[14].y - extension_threshold,  # Ring
        landmarks[20].y < landmarks[18].y - extension_threshold   # Pinky
    ]
    
    # Paper detection (all fingers extended and spread)
    if sum(fingers) == 5:
        # Check horizontal spread between fingers
        spread = (abs(index_tip.x - middle_tip.x) > 0.08 and
                 abs(middle_tip.x - ring_tip.x) > 0.08 and
                 abs(ring_tip.x - pinky_tip.x) > 0.08)
        if spread:
            return 'paper'
    
    # Scissors detection
    if fingers[0] and fingers[1] and not any(fingers[2:]):
        if abs(landmarks[8].y - landmarks[12].y) < extension_threshold:
            return 'scissors'
    
    # Rock detection
    if not thumb_open and not any(fingers):
        return 'rock'
    
    return 'unknown'

# Camera setup with forced parameters
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer to prevent lag

# Performance monitoring
performance_log = []
last_log_time = time.time()

try:
    while True:
        # Memory cleanup every 100 frames
        if len(performance_log) > 100:
            performance_log.clear()
            if cv2.getWindowProperty('RPS AI', 0) >= 0:
                cv2.destroyAllWindows()
            
        # Frame processing
        ret, frame = cap.read()
        if not ret:
            frame = frame_buffer
        
        # Efficient processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        gesture = 'unknown'
        process_time = 0
        
        if results.multi_hand_landmarks:
            classify_start = time.time()
            gesture = classify_gesture(results.multi_hand_landmarks[0].landmark)
            process_time = (time.time() - classify_start) * 1000
            
            # AI response mapping
            ai_move = {
                'rock': 'PAPER',
                'paper': 'SCISSORS',
                'scissors': 'ROCK'
            }.get(gesture, 'WAITING')
            
            # Display info (optimized)
            y_pos = 30
            cv2.putText(frame, f"YOU: {gesture}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(frame, f"AI: {ai_move}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
            cv2.putText(frame, f"{process_time:.1f}ms", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - last_frame_time + 1e-6)
        last_frame_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Performance logging
        if current_time - last_log_time > 1:
            print(f"Current FPS: {fps:.1f} | Process Time: {process_time:.1f}ms")
            last_log_time = current_time
        
        cv2.imshow('RPS AI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()