from flask import Flask, request, jsonify, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
import pickle

from zeroconf import ServiceInfo, Zeroconf
import socket

zeroconf = Zeroconf()

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)


ip = socket.inet_aton(ip_address)  # or your local IP
info = ServiceInfo(
    "_http._tcp.local.",
    "gesture._http._tcp.local.",
    addresses=[socket.inet_aton(ip_address)],
    port=8080,
)

zeroconf.register_service(info)

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
        body { font-family: Arial; text-align: center; margin-top: 50px; transition: background-color 0.3s; }
        button { padding: 20px; font-size: 24px; margin-top: 20px; cursor: pointer; }
        p { font-size: 28px; margin-top: 30px; }
        
        /* New styles for the big gesture display */
        #gestureDisplay {
            transition: all 0.3s ease;
        }
        .big-text {
            font-size: 100px !important;
            font-weight: bold;
            color: #4CAF50;
            display: block;
            margin-top: 40px;
            text-transform: uppercase;
        }
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
let isPaused = false; 
let buffer = [];
const alphaSmooth = 0.2;
let lastSample = {x:0,y:0,z:0,alpha:0,beta:0,gamma:0};

// --- New Endpointing Variables ---
let isMoving = false;
let quietFrames = 0;
const MOVEMENT_THRESHOLD = 0.3; // Must exceed this to start gesture (matches your crop threshold)
const QUIET_FRAMES_LIMIT = 20;  // How many frames of stillness before concluding the gesture is done
// ---------------------------------

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
    if(buffer.length < 10) {
        buffer = [];
        return; 
    }
    
    let processed = cropRecording(buffer);
    processed = resample(processed, 100);

    fetch("/predict", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({samples:processed})
    })
    .then(res=>res.json())
    .then(res=>{ 
        const gesture = res.predicted_gesture;
        display.textContent = gesture; 
        
        // Pause and show big text if it's a real gesture
        if (gesture && gesture.toLowerCase() !== "noise" && gesture !== "No data") {
            isPaused = true; 
            display.classList.add("big-text");
            
            setTimeout(() => {
                display.classList.remove("big-text");
                display.textContent = "None";
                buffer = []; 
                isPaused = false; 
            }, 500);
        }
    })
    .catch(err=>{ console.error(err); });
    
    buffer = [];
}

window.addEventListener("devicemotion", (event)=>{
    if(!permissionGranted || isPaused) return; 
    
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
    
    // Calculate motion magnitude (how hard the phone is moving)
    let mag = Math.sqrt(sample.x**2 + sample.y**2 + sample.z**2);

    if (!isMoving) {
        // Keep a small rolling buffer of 15 frames so we don't lose the very beginning of the movement
        buffer.push(sample);
        if (buffer.length > 15) buffer.shift();

        // If the movement spikes above threshold, lock in and start recording
        if (mag > MOVEMENT_THRESHOLD) {
            isMoving = true;
            quietFrames = 0;
        }
    } else {
        // We are currently actively recording a gesture
        buffer.push(sample);

        // Check if movement is dying down
        if (mag < MOVEMENT_THRESHOLD) {
            quietFrames++;
        } else {
            quietFrames = 0; // Reset quiet counter if movement spikes again
        }

        // If it's been quiet for enough frames, the gesture is officially done
        if (quietFrames >= QUIET_FRAMES_LIMIT) {
            sendBufferForPrediction(); 
            isMoving = false;
            quietFrames = 0;
        }
        
        // Failsafe: if the gesture takes way too long (e.g., someone shaking the phone forever), force a prediction
        if (buffer.length > 250) {
            sendBufferForPrediction();
            isMoving = false;
            quietFrames = 0;
        }
    }
});
</script>
</body>
</html>
"""

COMMAND_MAP =  {"flick_front": None,
                "flick_right": None,
                "flick_left": None,
                "flick_back": None,
                "noise": None}
def send_robot_command(gesture: str):
    if not COMMAND_MAP[gesture]:
        print("no known command for gesture: " + gesture)
        return
    COMMAND_MAP[gesture]()


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
    print(pred_label)
    if pred_label != "noise":
        send_robot_command(pred_label)

    return jsonify({"predicted_gesture": pred_label})

if __name__ == "__main__":
    # ssl_context='adhoc' generates a temporary certificate for HTTPS
    app.run(host="0.0.0.0", port=8080, ssl_context='adhoc')
