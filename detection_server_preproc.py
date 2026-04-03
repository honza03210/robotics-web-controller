from flask import Flask, request, jsonify, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
from robot import Robot
import pickle

app = Flask(__name__)

# Load model and label encoder
MODEL_FILE = "gesture_model.h5"
LE_FILE = "label_encoder.pkl"

print("Loading model...")
model = load_model(MODEL_FILE)
with open(LE_FILE, "rb") as f:
    label_encoder = pickle.load(f)
print("Model and label encoder loaded.")

# Robot setup
ROBOT = Robot()
COMMAND_MAP = {
    "flick_front": ROBOT.flick_front,
    "flick_right": ROBOT.flick_right,
    "flick_left": ROBOT.flick_left,
    "flick_back": ROBOT.flick_back,
    "noise": None
}

def send_robot_command(gesture: str, max_strength: float):
    max_strength = (max_strength + 0.5) ** 3
    if not COMMAND_MAP.get(gesture):
        print(f"No known command for gesture: {gesture} with power {max_strength}")
        return
    COMMAND_MAP[gesture](max_strength)

# HTML + JS
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Live Gesture Recognition</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* { box-sizing: border-box; margin:0; padding:0; }
html, body { font-family:"Segoe UI", Roboto, Arial; width:100%; height:100%; background:#121212; color:#fff; display:flex; justify-content:center; align-items:center; }
.card { width:100%; height:100vh; padding:6vw; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; }
h1 { font-size:8vw; margin-bottom:3vw; }
p { font-size:4.5vw; opacity:0.8; margin-top:2vw; }
button { margin-top:5vw; padding:5vw 0; width:100%; max-width:400px; font-size:5vw; font-weight:600; border:none; border-radius:14px; background:#1f1f1f; color:#00ffcc; cursor:pointer; transition:0.2s; }
button:hover { transform:scale(1.02); } button:active { transform:scale(0.98); }
#status { margin-top:3vw; font-size:4vw; opacity:0.7; }
#gestureDisplay { margin-top:6vw; font-size:12vw; font-weight:800; color:#00ffcc; transition:all 0.2s ease; }
.big-text { font-size:18vw !important; color:#00ffcc; }
#motionStats { margin-top:3vw; font-size:4vw; opacity:0.7; }
#probDisplay {
    margin-top:4vw;
    width:100%;
    max-width:450px;
}

.prob-bar {
    display:flex;
    align-items:center;
    margin-bottom:10px;
    background:#1e1e1e;
    border-radius:8px;
    padding:6px 10px;
}

.prob-label {
    width:35%;
    font-size:3.5vw;
}

.prob-track {
    flex-grow:1;
    height:10px;
    background:#333;
    border-radius:6px;
    overflow:hidden;
    margin-left:10px;
}

.prob-fill {
    height:100%;
    background:linear-gradient(90deg,#00ffcc,#00aaff);
    width:0%;
    transition:width 0.4s ease;
}

.prob-value {
    width:50px;
    text-align:right;
    font-size:3vw;
    margin-left:8px;
}

.predicted {
    background:#003d35;
}
</style>
</head>
<body>
<div class="card">
<h1>Live Gesture Recognition</h1>
<p>Enable motion sensors and move your device!</p>
<button id="enableSensorsBtn">Enable Motion Sensors</button>
<p id="status"></p>
<p>Predicted Gesture:</p>
<span id="gestureDisplay">None</span>
<div id="probDisplay"></div>
<p id="motionStats"></p>
</div>

<script>
let permissionGranted=false, isPaused=false, buffer=[];
const alphaSmooth=0.2, lastSample={x:0,y:0,z:0,alpha:0,beta:0,gamma:0};
let isMoving=false, quietFrames=0;
const MOVEMENT_THRESHOLD=0.3, QUIET_FRAMES_LIMIT=20;

const status=document.getElementById("status");
const display=document.getElementById("gestureDisplay");
const motionStats=document.getElementById("motionStats");
const probDisplay=document.getElementById("probDisplay");
const enableBtn=document.getElementById("enableSensorsBtn");

enableBtn.addEventListener("click", async()=>{
    if(typeof DeviceMotionEvent!=="undefined" && typeof DeviceMotionEvent.requestPermission==="function"){
        try{
            const resp=await DeviceMotionEvent.requestPermission();
            if(resp==="granted"){ permissionGranted=true; status.textContent="Motion sensors enabled ✅"; enableBtn.style.display="none"; }
            else status.textContent="Permission denied ❌";
        }catch(err){ console.error(err); status.textContent="Error requesting permission"; }
    } else { permissionGranted=true; status.textContent="Motion sensors available ✅"; enableBtn.style.display="none"; }
});

function correctAxes(s){ let {x,y,z}=s; if(/Android/i.test(navigator.userAgent)){ let tmp=y; y=-z; z=tmp; } return {x,y,z, alpha:s.alpha, beta:s.beta, gamma:s.gamma}; }
function normalizeSample(s){ return {x:s.x/20, y:s.y/20, z:s.z/20, alpha:s.alpha/200, beta:s.beta/200, gamma:s.gamma/200}; }
function smoothSample(s){ let sm={}; for(let k in s){ sm[k]=alphaSmooth*s[k]+(1-alphaSmooth)*lastSample[k]; lastSample[k]=sm[k]; } return sm; }

function cropRecording(samples, threshold=0.3, padding=15){
    let start=0,end=samples.length-1;
    for(let i=0;i<samples.length;i++){ let mag=Math.sqrt(samples[i].x**2+samples[i].y**2+samples[i].z**2); if(mag>threshold){ start=Math.max(0,i-padding); break; } }
    for(let i=samples.length-1;i>=0;i--){ let mag=Math.sqrt(samples[i].x**2+samples[i].y**2+samples[i].z**2); if(mag>threshold){ end=Math.min(samples.length-1,i+padding); break; } }
    return samples.slice(start,end+1);
}

function resample(samples,targetLength=100){
    if(samples.length===targetLength) return samples;
    let resampled=[];
    for(let i=0;i<targetLength;i++){
        let idx=i*(samples.length-1)/(targetLength-1);
        let low=Math.floor(idx), high=Math.ceil(idx), t=idx-low, s={};
        ["x","y","z","alpha","beta","gamma"].forEach(k=>{ s[k]=samples[low][k]*(1-t)+samples[high][k]*t; });
        resampled.push(s);
    }
    return resampled;
}

function displayProbabilities(probDict){

    probDisplay.innerHTML="";

    const sorted = Object.entries(probDict)
        .sort((a,b)=>b[1]-a[1]);

    const bestLabel = sorted[0][0];

    sorted.forEach(([label,prob])=>{

        const percent = (prob*100).toFixed(1);

        const barDiv=document.createElement("div");
        barDiv.className="prob-bar";

        if(label===bestLabel){
            barDiv.classList.add("predicted");
        }

        barDiv.innerHTML=`
            <span class="prob-label">${label}</span>
            <div class="prob-track">
                <div class="prob-fill" style="width:${percent}%"></div>
            </div>
            <span class="prob-value">${percent}%</span>
        `;

        probDisplay.appendChild(barDiv);
    });
}

function sendBufferForPrediction(){
    if(buffer.length<10){ buffer=[]; return; }
    let cropped=cropRecording(buffer);
    let motionLength=cropped.length;
    let processed=resample(cropped,100);
    let strengths=processed.map(s=>Math.sqrt(s.x**2+s.y**2+s.z**2));
    let maxStrength=Math.max(...strengths);
    let avgStrength=strengths.reduce((a,b)=>a+b,0)/strengths.length;
    motionStats.textContent=`Max: ${maxStrength.toFixed(3)} | Avg: ${avgStrength.toFixed(3)} | Frames: ${motionLength}`;

    fetch("/predict",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({samples:processed,max_strength:maxStrength,avg_strength:avgStrength,motion_length:motionLength})
    })
    .then(res=>res.json())
    .then(res=>{
        display.textContent=res.predicted_gesture;
        displayProbabilities(res.probabilities);

        if(res.predicted_gesture && res.predicted_gesture.toLowerCase()!=="noise" && res.predicted_gesture!=="No data"){
            isPaused=true;
            display.classList.add("big-text");
            setTimeout(()=>{
                display.classList.remove("big-text");
                display.textContent="None";
                probDisplay.innerHTML="";
                buffer=[];
                isPaused=false;
            },2000);
        }
    })
    .catch(err=>console.error(err));
    buffer=[];
}

window.addEventListener("devicemotion", event=>{
    if(!permissionGranted || isPaused) return;
    let sample={x:event.acceleration.x||0,y:event.acceleration.y||0,z:event.acceleration.z||0,
                alpha:event.rotationRate.alpha||0,beta:event.rotationRate.beta||0,gamma:event.rotationRate.gamma||0};
    sample=correctAxes(sample);
    sample=normalizeSample(sample);
    sample=smoothSample(sample);

    let mag=Math.sqrt(sample.x**2+sample.y**2+sample.z**2);
    if(!isMoving){
        buffer.push(sample);
        if(buffer.length>15) buffer.shift();
        if(mag>MOVEMENT_THRESHOLD){ isMoving=true; quietFrames=0; }
    } else {
        buffer.push(sample);
        if(mag<MOVEMENT_THRESHOLD){ quietFrames++; } else { quietFrames=0; }
        if(quietFrames>=QUIET_FRAMES_LIMIT || buffer.length>250){ sendBufferForPrediction(); isMoving=false; quietFrames=0; }
    }
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
        return jsonify({"predicted_gesture": "No data", "probabilities": {}}), 400

    max_strength = content.get("max_strength", 0)

    X = np.array([[s["x"], s["y"], s["z"], s["alpha"], s["beta"], s["gamma"]] for s in samples]).flatten()[np.newaxis, :]
    pred_probs = model.predict(X, verbose=0)[0]
    pred_label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
    prob_dict = {label: float(prob) for label, prob in zip(label_encoder.classes_, pred_probs)}

    print(f"Gesture: {pred_label} | Max strength: {max_strength:.3f} | Probs: {prob_dict}")

    if pred_label != "noise":
        send_robot_command(pred_label, max_strength)

    return jsonify({"predicted_gesture": pred_label, "probabilities": prob_dict})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, ssl_context='adhoc')