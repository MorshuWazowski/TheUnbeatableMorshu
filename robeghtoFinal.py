import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load original dataset
df = pd.read_csv('final_training_dataset.csv')

# Drop Z columns before doing anything else
z_cols = [f"z{i}" for i in range(1, 22)]
df = df.drop(columns=z_cols)

# Normalize only x and y
def normalize_hand_landmarks(row):
    x_wrist, y_wrist = row["x1"], row["y1"]
    x_tip, y_tip = row["x9"], row["y9"]

    scale = np.sqrt((x_tip - x_wrist) ** 2 + (y_tip - y_wrist) ** 2)
    if scale == 0:
        return row

    normalized_row = row.copy()
    for i in range(1, 22):
        normalized_row[f"x{i}"] = (row[f"x{i}"] - x_wrist) / scale
        normalized_row[f"y{i}"] = (row[f"y{i}"] - y_wrist) / scale

    return normalized_row

# Apply normalization (only on X/Y)
xy_cols = [col for col in df.columns if col.startswith(('x', 'y'))]
df[xy_cols] = df[xy_cols].apply(normalize_hand_landmarks, axis=1)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split
X = df.drop(['label'], axis=1)
y = df['label']

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
xgb_classifier = XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42
)

xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate
y_pred_train = xgb_classifier.predict(X_train_scaled)
y_pred_test = xgb_classifier.predict(X_test_scaled)

print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test) * 100:.2f}%")

# Save everything
joblib.dump(xgb_classifier, "xgboost_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Model and label encoder saved (with Zs dropped).")
