import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm

# Constants (1-indexed)
LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
EXPECTED_COLUMNS = [f'{ax}{i}' for i in range(1, 22) for ax in ['x', 'y', 'z']] + ['label']

def process_hagrid_data(hagrid_path):
    """Process HaGRID dataset with 1-indexed columns"""
    df = pd.read_csv(hagrid_path)
    
    # 1. Filter and rename labels
    label_map = {'fist': 'rock', 'palm': 'paper', 'peace': 'scissors', 'rock': 'rock'}
    df = df[df['label'].isin(label_map.keys())].copy()
    df['label'] = df['label'].map(label_map)
    
    # 2. Verify 1-indexed columns
    required_columns = [f'{ax}{i}' for i in range(1,22) for ax in ['x','y','z']] + ['label']
    missing = set(required_columns) - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing columns in HaGRID data: {missing}")
    
    return df[EXPECTED_COLUMNS].dropna()

def process_new_images(image_dir):
    """Process new images with 1-indexed output"""
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    data = []
    for label in ['rock', 'paper', 'scissors']:
        label_dir = os.path.join(image_dir, label)
        if not os.path.exists(label_dir):
            continue
            
        images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(images, desc=f'Processing {label}'):
            img_path = os.path.join(label_dir, img_file)
            row = process_single_image(img_path, label, mp_hands)
            if row:
                data.append(row)
    
    mp_hands.close()
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS) if data else pd.DataFrame()

def process_single_image(img_path, label, hands):
    """Process image with 1-indexed landmark output"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
            
        # Convert MediaPipe's 0-index to our 1-index
        landmarks = []
        for idx, lm in enumerate(results.multi_hand_landmarks[0].landmark, 1):
            landmarks.extend([lm.x, lm.y, lm.z])
            
        return landmarks + [label.strip().lower()]
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def combine_datasets(old_data, new_data):
    """Combine datasets with strict 1-index validation"""
    combined = pd.concat([old_data, new_data], ignore_index=True)
    
    # Validate columns
    if list(combined.columns) != EXPECTED_COLUMNS:
        raise ValueError("Column mismatch after combining datasets!")
    
    # Final cleaning
    combined = combined.dropna()
    combined = combined[combined['label'].isin(['rock', 'paper', 'scissors'])]
    return combined

def main():
    # Process old HaGRID data
    hagrid_path = r'C:\Users\malek\Downloads\hand_landmarks_data.csv'
    if os.path.exists(hagrid_path):
        hagrid_df = process_hagrid_data(hagrid_path)
        print(f"Processed HaGRID data: {len(hagrid_df)} rows")
    else:
        hagrid_df = pd.DataFrame()
        print("No HaGRID data found")

    # Process new images
    image_dir = r'C:\Users\malek\Downloads\RPS\train'
    new_df = process_new_images(image_dir)
    print(f"Processed new data: {len(new_df)} rows")

    # Combine datasets
    final_df = combine_datasets(hagrid_df, new_df)
    print(f"Final dataset size: {len(final_df)} rows")
    print("Label distribution:\n", final_df['label'].value_counts())

    # Save results
    final_df.to_csv('final_dataset.csv', index=False)
    print("Dataset saved successfully")

if __name__ == "__main__":
    main()