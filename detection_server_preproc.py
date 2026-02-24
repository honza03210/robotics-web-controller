from flask import Flask, request, jsonify, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load trained model and label encoder
MODEL_FILE = "gesture_model.h5"
LE_FILE = "label_encoder.pkl"

print("Loading model...")
model = load_model(MODEL_FILE)
with open(LE_FILE, "rb") as f:
    label_encoder = pickle.load(f)
print("Model and label encoder loaded.")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Gesture Recognition</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        button { padding: 20px; font-size: 24px; margin-top: 20px; }
        p { font-size: 28px; margin-top: 30px; }
    </style>
</head>
<body>
    <h1>Live Gesture Recognition</h1>
    <p>Enable motion sensors and move your device!</p>
    <button id="enableSensorsBtn">Enable Motion Sensors</button>
    <p id="status"></p>
    <p>Predicted Gesture: <span id="gestureDisplay">None</span></p>

<script>
let permissionGranted = false;
let buffer = [];
const alphaSmooth = 0.2;
let lastSample = {x:0,y:0,z:0,alpha:0,beta:0,gamma:0};
const status = document.getElementById("status");
const display = document.getElementById("gestureDisplay");
const enableBtn = document.getElementById("enableSensorsBtn");

// iOS motion permission
enableBtn.addEventListener("click", async () => {
    if (typeof DeviceMotionEvent !== "undefined" &&
        typeof DeviceMotionEvent.requestPermission === "function") {
        try {
            const response = await DeviceMotionEvent.requestPermission();
            if (response === "granted") {
                permissionGranted = true;
                status.textContent = "Motion sensors enabled ✅";
                enableBtn.style.display = "none";
            } else {
                status.textContent = "Permission denied ❌";
            }
        } catch (err) {
            console.error(err);
            status.textContent = "Error requesting permission";
        }
    } else {
        permissionGranted = true;
        status.textContent = "Motion sensors available ✅";
        enableBtn.style.display = "none";
    }
});

// Preprocessing
function correctAxes(sample) {
    let {x,y,z} = sample;
    if (/Android/i.test(navigator.userAgent)) {
        let tmp = y;
        y = -z;
        z = tmp;
    }
    return {x,y,z, alpha: sample.alpha, beta: sample.beta, gamma: sample.gamma};
}

function normalizeSample(sample) {
    return {
        x: sample.x / 20,
        y: sample.y / 20,
        z: sample.z / 20,
        alpha: sample.alpha / 200,
        beta: sample.beta / 200,
        gamma: sample.gamma / 200
    };
}

function smoothSample(sample) {
    let smoothed = {};
    for (let key in sample) {
        smoothed[key] = alphaSmooth*sample[key] + (1-alphaSmooth)*lastSample[key];
        lastSample[key] = smoothed[key];
    }
    return smoothed;
}

function cropRecording(samples, threshold=0.3, padding=15) {
    let start=0, end=samples.length-1;
    for(let i=0;i<samples.length;i++){
        let mag = Math.sqrt(samples[i].x**2 + samples[i].y**2 + samples[i].z**2);
        if(mag > threshold){ start=Math.max(0,i-padding); break; }
    }
    for(let i=samples.length-1;i>=0;i--){
        let mag = Math.sqrt(samples[i].x**2 + samples[i].y**2 + samples[i].z**2);
        if(mag > threshold){ end=Math.min(samples.length-1,i+padding); break; }
    }
    return samples.slice(start,end+1);
}

function resample(samples, targetLength=100){
    if(samples.length === targetLength) return samples;
    let resampled = [];
    for(let i=0;i<targetLength;i++){
        let idx = i*(samples.length-1)/(targetLength-1);
        let low = Math.floor(idx);
        let high = Math.ceil(idx);
        let t = idx - low;
        let s = {};
        ["x","y","z","alpha","beta","gamma"].forEach(key=>{
            let v = samples[low][key]*(1-t) + samples[high][key]*t;
            s[key] = v;
        });
        resampled.push(s);
    }
    return resampled;
}

// Capture live motion
function sendBufferForPrediction(){
    if(buffer.length<10) return;
    let processed = cropRecording(buffer);
    processed = resample(processed, 100);

    fetch("/predict", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({samples:processed})
    })
    .then(res=>res.json())
    .then(res=>{ display.textContent = res.predicted_gesture; })
    .catch(err=>{ console.error(err); });
    buffer = [];
}

window.addEventListener("devicemotion", (event)=>{
    if(!permissionGranted) return;
    let sample = {
        x: event.acceleration.x || 0,
        y: event.acceleration.y || 0,
        z: event.acceleration.z || 0,
        alpha: event.rotationRate.alpha || 0,
        beta: event.rotationRate.beta || 0,
        gamma: event.rotationRate.gamma || 0
    };
    sample = correctAxes(sample);
    sample = normalizeSample(sample);
    sample = smoothSample(sample);
    buffer.push(sample);
    if(buffer.length>=50){ sendBufferForPrediction(); }
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    content = request.get_json()
    samples = content.get("samples")
    if not samples:
        return jsonify({"predicted_gesture":"No data"}), 400

    # Flatten 100 samples x 6 features -> 600-dim vector
    X = np.array([[s["x"],s["y"],s["z"],s["alpha"],s["beta"],s["gamma"]] for s in samples]).flatten()[np.newaxis,:]
    
    # Predict
    pred_probs = model.predict(X, verbose=0)
    pred_label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
    return jsonify({"predicted_gesture": pred_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6666)