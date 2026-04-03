import json
import numpy as np
import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# File paths
DATA_FILE = "gesture_data_resampled.json"
MODEL_FILE = "gesture_model.h5"

# Parameters
INPUT_TIME_STEPS = 100
INPUT_FEATURES = 6  # x, y, z, alpha, beta, gamma
AUGMENTATION_FACTOR = 3  # Increase dataset size

# 1️⃣ Load dataset
with open(DATA_FILE, "r") as f:
    data = json.load(f)

X = []
y = []

for entry in data:
    gesture = entry["gesture"]
    samples = entry["samples"]

    flat_sample = []
    for s in samples:
        flat_sample.extend([
            s["x"], s["y"], s["z"],
            s["alpha"], s["beta"], s["gamma"]
        ])

    X.append(flat_sample)
    y.append(gesture)

X = np.array(X)
y = np.array(y)

print(f"Original dataset: {X.shape[0]} samples")

# 2️⃣ Data augmentation (CRITICAL for small datasets)
def augment_data(X, y, factor=2):
    X_aug = []
    y_aug = []

    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])

        for _ in range(factor):
            noise = np.random.normal(0, 0.02, X[i].shape)
            scaled = X[i] * np.random.uniform(0.95, 1.05)

            augmented = scaled + noise

            X_aug.append(augmented)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)

X, y = augment_data(X, y, AUGMENTATION_FACTOR)
print(f"Augmented dataset: {X.shape[0]} samples")

# 3️⃣ Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4️⃣ Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 5️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 6️⃣ Build SMALLER model (better for small data)
model = Sequential([
    Dense(64, input_dim=INPUT_TIME_STEPS * INPUT_FEATURES, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(y_categorical.shape[1], activation='softmax')
])

# 7️⃣ Compile (less smoothing for small data)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=['accuracy']
)

model.summary()

# 8️⃣ Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# 9️⃣ Train (smaller batch = better generalization)
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# 🔟 Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# 11️⃣ Save model
model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# 12️⃣ Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# 13️⃣ Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Label encoder saved to label_encoder.pkl")
print("Scaler saved to scaler.pkl")