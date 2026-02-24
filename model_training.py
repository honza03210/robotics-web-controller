import json
import numpy as np
import tensorflowjs as tfjs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# File paths
DATA_FILE = "gesture_data_resampled.json"
MODEL_FILE = "gesture_model.h5"

# Parameters
INPUT_TIME_STEPS = 100
INPUT_FEATURES = 6  # x, y, z, alpha, beta, gamma

# 1️⃣ Load dataset
with open(DATA_FILE, "r") as f:
    data = json.load(f)

X = []
y = []

for entry in data:
    gesture = entry["gesture"]
    samples = entry["samples"]  # list of dicts
    # Flatten time series: 100 samples × 6 features → 600 features
    flat_sample = []
    for s in samples:
        flat_sample.extend([s["x"], s["y"], s["z"], s["alpha"], s["beta"], s["gamma"]])
    X.append(flat_sample)
    y.append(gesture)

X = np.array(X)
y = np.array(y)

print(f"Dataset: {X.shape[0]} samples, input dimension {X.shape[1]}")

# 2️⃣ Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 3️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# 4️⃣ Build simple fully connected model
model = Sequential([
    Dense(256, input_dim=INPUT_TIME_STEPS*INPUT_FEATURES, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5️⃣ Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

# 6️⃣ Save model
model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# 7️⃣ Save label encoder for inference
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved to label_encoder.pkl")