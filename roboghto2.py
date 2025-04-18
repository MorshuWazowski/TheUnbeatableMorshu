# Add these imports at the top of your script
import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\malek\Downloads\rps_model.h5')
# Class labels (adjust based on your training data)
class_names = ['paper', 'rock', 'scissors']  # Check train_generator.class_indices

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame (resize, normalize)
    roi = cv2.resize(frame, (150, 150))  # Match your model's input size (150x150 or 256x256)
    roi = roi / 255.0  # Normalize pixel values
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension: (1, 150, 150, 3)
    
    # Predict
    prediction = model.predict(roi)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display prediction
    cv2.putText(frame, f"Move: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Rock-Paper-Scissors', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()