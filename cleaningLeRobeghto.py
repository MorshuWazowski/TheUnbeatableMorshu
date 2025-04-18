import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import acos, degrees
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Helper functions
def distance(a, b):
    return np.linalg.norm(a - b)

def apply_augmentations(image):
    h, w = image.shape[:2]
    
    # Random horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # Random rotation
    angle = random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random translation
    tx = random.uniform(-0.05, 0.05) * w
    ty = random.uniform(-0.05, 0.05) * h
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random shearing
    shear_factor = random.uniform(-0.15, 0.15)
    M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    image = cv2.warpAffine(image, M_shear, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Optional slight zoom
    scale = random.uniform(0.9, 1.1)
    M_zoom = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    image = cv2.warpAffine(image, M_zoom, (w, h), borderMode=cv2.BORDER_REFLECT)

    return image

def angle(a, b, c):
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_landmarks(image):
    if image is None or image.size == 0:
        return None
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark
    return None

def compute_features(landmarks):
    points = np.array([[lm.x, lm.y] for lm in landmarks])

    wrist = points[0]
    tips = [4, 8, 12, 16, 20]  # thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
    mcp = [2, 5, 9, 13, 17]
    pip = [3, 6, 10, 14, 18]

    # Distances from wrist to fingertips
    dist_wrist_fingertips = [distance(wrist, points[i]) for i in tips]

    # Spread between adjacent fingertips (ignoring thumb)
    spreads = [distance(points[tips[i]], points[tips[i+1]]) for i in range(1, 4)]

    # Finger curl (tip to mcp / full finger length)
    curls = [
        distance(points[tips[i]], points[mcp[i]]) / distance(points[mcp[i]], wrist)
        for i in range(5)
    ]

    # Angles at PIP joints
    joint_angles = [angle(points[mcp[i]], points[pip[i]], points[tips[i]]) for i in range(5)]

    # Hand shape metrics
    hand_width = distance(points[4], points[20])
    hand_length = distance(wrist, points[12])
    aspect_ratio = hand_width / hand_length if hand_length != 0 else 0

    # Binary flags
    is_hand_open = int(all(distance(points[i], wrist) > 0.1 for i in tips[1:]))  # fingers extended
    thumb_crossed = int(points[4][0] < points[3][0])  # thumb crosses palm

    return dist_wrist_fingertips + spreads + curls + joint_angles + [hand_width, aspect_ratio, is_hand_open, thumb_crossed]

def process_folder(folder_path, label, augment_count=2):
    data = []
    for fname in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, fname)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue
            for _ in range(augment_count + 1):  # original + augmentations
                aug_img = apply_augmentations(image) if _ > 0 else image
                landmarks = extract_landmarks(aug_img)
                if landmarks:
                    features = compute_features(landmarks)
                    data.append(features + [label])
                else:
                    print(f"❌ No landmarks detected in {fname} (aug { _ })")
    return pd.DataFrame(data)

# Main process
base_train = r'C:\Users\malek\Downloads\RPS\test'

df_rock = process_folder(os.path.join(base_train, 'rock'), 'rock')
df_paper = process_folder(os.path.join(base_train, 'paper'), 'paper')
df_scissors = process_folder(os.path.join(base_train, 'scissors'), 'scissors')

# Define column names
columns = [
    'dist_wrist_thumbtip', 'dist_wrist_indextip', 'dist_wrist_midtip', 'dist_wrist_ringtip', 'dist_wrist_pinkytip',
    'spread_index_middle', 'spread_middle_ring', 'spread_ring_pinky',
    'curl_thumb', 'curl_index', 'curl_middle', 'curl_ring', 'curl_pinky',
    'angle_thumb', 'angle_index', 'angle_middle', 'angle_ring', 'angle_pinky',
    'hand_width', 'aspect_ratio', 'is_hand_open', 'thumb_crossed', 'label'
]

# Combine and save
all_data = pd.concat([df_rock, df_paper, df_scissors])
all_data.columns = columns
all_data.to_csv('rps_train_full_features.csv', index=False)
print("✅ Feature-rich CSV saved as rps_train_full_features.csv")
