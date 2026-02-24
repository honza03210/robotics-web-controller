from flask import Flask, request, jsonify, render_template_string, send_file
import json
import os

app = Flask(__name__)
DATA_FILE = "gesture_data.json"

# Load existing data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        gesture_data = json.load(f)
else:
    gesture_data = []

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gesture Data Collector</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        button { padding: 20px; font-size: 24px; margin-top: 20px; margin:5px; }
        select { font-size: 20px; padding: 5px; }
    </style>
</head>
<body>
    <h1>Gesture Data Collector</h1>

    <p>Step 1: Enable motion sensors (required for iOS)</p>
    <button id="enableSensorsBtn">Enable Motion Sensors</button>
    <p id="status"></p>

    <p>Step 2: Select gesture to record:</p>
    <select id="gestureSelect">
        <option value="flick_front">Flick Front</option>
        <option value="flick_right">Flick Side Right</option>
        <option value="flick_left">Flick Side Left</option>
        <option value="flick_back">Flick Side Back</option>
        <option value="noise">Do Nothing / Noise</option>
    </select>
    <br><br>

    <button id="recordBtn">Tap to Record 3s</button>
    <button id="submitBtn">Submit All Data</button>
    <p id="recordStatus"></p>

    <p>Step 3: Download dataset when done:</p>
    <a href="/download"><button>Download JSON Dataset</button></a>

<script>
let permissionGranted = false;
let buffer = [];
let batchData = [];
const recordDuration = 3000; // 3 seconds
const alphaSmooth = 0.2;
let lastSample = {x:0,y:0,z:0,alpha:0,beta:0,gamma:0};

const status = document.getElementById("status");
const recordStatus = document.getElementById("recordStatus");
const enableBtn = document.getElementById("enableSensorsBtn");
const gestureSelect = document.getElementById("gestureSelect");
const recordBtn = document.getElementById("recordBtn");
const submitBtn = document.getElementById("submitBtn");

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

// Cropping based on acceleration magnitude
function cropRecording(samples, threshold=0.3, padding=15) {
    let start=0, end=samples.length-1;

    // If gesture is "noise", skip cropping
    const gesture = gestureSelect.value;
    if(gesture === "noise") return samples;

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

// Record 3s
recordBtn.addEventListener("click", () => {
    if (!permissionGranted) { recordStatus.textContent = "Enable motion sensors first!"; return; }

    buffer=[];
    const gesture = gestureSelect.value;
    recordStatus.textContent = `Recording "${gesture}" for 3 seconds...`;

    function recordMotion(event){
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
        sample.timestamp = Date.now();
        buffer.push(sample);
    }

    window.addEventListener("devicemotion", recordMotion);

    setTimeout(()=>{
        window.removeEventListener("devicemotion", recordMotion);
        buffer = cropRecording(buffer);
        batchData.push({gesture:gesture, samples:buffer});
        recordStatus.textContent = `Recorded ${buffer.length} preprocessed samples for "${gesture}". Total in batch: ${batchData.length}`;
    }, recordDuration);
});

// Submit batch
submitBtn.addEventListener("click", ()=>{
    if(batchData.length===0){ recordStatus.textContent="No recordings to submit!"; return; }

    fetch("/save_batch", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({batch:batchData})
    })
    .then(res=>res.json())
    .then(res=>{ recordStatus.textContent=res.message; batchData=[]; })
    .catch(err=>{ console.error(err); recordStatus.textContent="Error submitting batch"; });
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/save_batch", methods=["POST"])
def save_batch():
    global gesture_data
    content = request.get_json()
    batch = content.get("batch")
    if not batch:
        return jsonify({"message":"No data received"}), 400

    gesture_data.extend(batch)

    with open(DATA_FILE, "w") as f:
        json.dump(gesture_data, f, indent=2)

    return jsonify({"message": f"Saved batch of {len(batch)} recordings successfully."})

@app.route("/download")
def download():
    return send_file(DATA_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6666)