from flask import Flask, send_from_directory, render_template_string

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Client-Side Gesture Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.12.0/dist/tf.min.js"></script>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        p { font-size: 20px; }
    </style>
</head>
<body>
    <h1>Client-Side Gesture Detection</h1>
    <button id="enableBtn">Enable Motion Sensors</button>
    <p id="status">Loading model...</p>
    <p>Detected Gesture: <span id="gesture">None</span></p>

<script>
let permissionGranted = false;
let buffer = [];
const WINDOW_SIZE = 100; // must match training
const alphaSmooth = 0.2;
let lastSample = {x:0,y:0,z:0,alpha:0,beta:0,gamma:0};
let detecting = true;
let displayResetTimeout = null;
const status = document.getElementById("status");
const gestureLabel = document.getElementById("gesture");
const enableBtn = document.getElementById("enableBtn");

let model, labels = ["flick_front","flick_right","flick_left","flick_back","noise"]; // adjust to your training

// Load the model
(async () => {
    model = await tf.loadLayersModel("/tfjs_model/model.json");
    status.textContent = "Model loaded ✅";
})();

// Enable motion sensors
enableBtn.addEventListener("click", async () => {
    if (typeof DeviceMotionEvent.requestPermission === "function") {
        const response = await DeviceMotionEvent.requestPermission();
        if(response==="granted"){ permissionGranted=true; status.textContent="Motion sensors enabled ✅"; enableBtn.style.display="none"; }
        else{ status.textContent="Permission denied ❌"; }
    } else { permissionGranted=true; status.textContent="Motion sensors available ✅"; enableBtn.style.display="none"; }
});

// Preprocessing functions
function correctAxes(s){ let {x,y,z}=s; if(/Android/i.test(navigator.userAgent)){let tmp=y;y=-z;z=tmp;} return {x,y,z,alpha:s.alpha,beta:s.beta,gamma:s.gamma}; }
function normalizeSample(s){return {x:s.x/20,y:s.y/20,z:s.z/20,alpha:s.alpha/200,beta:s.beta/200,gamma:s.gamma/200};}
function smoothSample(s){let sm={};for(let k in s){sm[k]=alphaSmooth*s[k]+(1-alphaSmooth)*lastSample[k];lastSample[k]=sm[k];}return sm;}
function cropRecording(samples, threshold=0.3, padding=10){let start=0,end=samples.length-1;for(let i=0;i<samples.length;i++){let mag=Math.sqrt(samples[i].x**2+samples[i].y**2+samples[i].z**2);if(mag>threshold){start=Math.max(0,i-padding);break;}}for(let i=samples.length-1;i>=0;i--){let mag=Math.sqrt(samples[i].x**2+samples[i].y**2+samples[i].z**2);if(mag>threshold){end=Math.min(samples.length-1,i+padding);break;}}return samples.slice(start,end+1);}

// Real-time monitoring
window.addEventListener("devicemotion", async (event)=>{
    if(!permissionGranted || !detecting || !model) return;

    let s = {x:event.acceleration.x||0,y:event.acceleration.y||0,z:event.acceleration.z||0,
             alpha:event.rotationRate.alpha||0,beta:event.rotationRate.beta||0,gamma:event.rotationRate.gamma||0};
    s = correctAxes(s); s = normalizeSample(s); s = smoothSample(s);
    buffer.push(s); if(buffer.length>WINDOW_SIZE) buffer.shift();

    if(buffer.length===WINDOW_SIZE){
        let windowData = cropRecording(buffer);
        if(windowData.length<WINDOW_SIZE) return;

        const input = tf.tensor([].concat(...windowData.map(s=>[s.x,s.y,s.z,s.alpha,s.beta,s.gamma]))).reshape([1,WINDOW_SIZE*6]);
        const pred = model.predict(input);
        const probs = await pred.data();
        const maxIdx = probs.indexOf(Math.max(...probs));
        const gesture = labels[maxIdx];
        const prob = probs[maxIdx];

        if(prob>=0.7){
            gestureLabel.textContent = gesture + " (" + (prob*100).toFixed(1)+"%)";
            detecting=false;
            buffer=[];
            if(displayResetTimeout) clearTimeout(displayResetTimeout);
            displayResetTimeout = setTimeout(()=>{ gestureLabel.textContent="None"; detecting=true; }, 500);
        }
        tf.dispose([input,pred]);
    }
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# Serve TF.js model files
@app.route("/tfjs_model/<path:path>")
def serve_model(path):
    return send_from_directory("tfjs_model", path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6666)