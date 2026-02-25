import numpy as np
import pickle
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file

# -------------------------
# CONFIG
# -------------------------
MODEL_FILE = "gesture_model.keras"          # your trained model
LABEL_ENCODER_FILE = "label_encoder.pkl"   # your label encoder
INPUT_TIME_STEPS = 100
INPUT_FEATURES = 6  # x, y, z, alpha, beta, gamma

# -------------------------
# LOAD MODEL AND LABEL ENCODER
# -------------------------
print("Loading model...")
model = load_model(MODEL_FILE)
print("Model loaded.")

with open(LABEL_ENCODER_FILE, "rb") as f:
    le = pickle.load(f)
print("Label encoder loaded.")

# -------------------------
# CREATE FLASK APP
# -------------------------
app = Flask(__name__)

# Serve the HTML page
@app.route("/")
def index():
    return send_file("gesture_client.html")  # see HTML below

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload:
    {
        "samples": [
            {"x":..,"y":..,"z":..,"alpha":..,"beta":..,"gamma":..},
            ... 100 samples total ...
        ]
    }
    Returns JSON with predicted gesture and probabilities.
    """
    data = request.json
    samples = data.get("samples")
    
    # Validate input
    if samples is None or len(samples) != INPUT_TIME_STEPS:
        return jsonify({"error": f"Expected {INPUT_TIME_STEPS} samples"}), 400

    # Flatten samples: 100x6 → 600 features
    flat_input = []
    for s in samples:
        flat_input.extend([
            s.get("x", 0),
            s.get("y", 0),
            s.get("z", 0),
            s.get("alpha", 0),
            s.get("beta", 0),
            s.get("gamma", 0)
        ])
    
    x_input = np.array([flat_input])  # shape (1, 600)
    
    # Predict probabilities
    preds = model.predict(x_input)
    
    # Get label with highest probability
    pred_class_index = np.argmax(preds, axis=1)[0]
    pred_label = le.inverse_transform([pred_class_index])[0]

    return jsonify({
        "prediction": pred_label,
        "probabilities": preds[0].tolist()
    })

# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:6666")
    app.run(host="0.0.0.0", port=6666, debug=True)