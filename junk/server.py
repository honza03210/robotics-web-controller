from flask import Flask, request, session, redirect, url_for, render_template_string, jsonify
import logging

app = Flask(__name__)
app.secret_key = "CHANGE_THIS_SECRET_KEY"

PASSWORD = "mysecretpassword"

logging.basicConfig(level=logging.INFO)

# HTML page with WASD controls
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>WASD Controller</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 40px; }
        .key { font-size: 40px; padding: 20px; margin: 10px; display: inline-block; border: 2px solid #333; width: 80px; }
    </style>
</head>
<body>
    <h1>WASD Control Panel</h1>
    <p>Use your keyboard (W A S D)</p>
    <div>
        <div class="key">W</div><br>
        <div class="key">A</div>
        <div class="key">S</div>
        <div class="key">D</div>
    </div>

<script>
document.addEventListener("keydown", function(e) {
    let key = e.key.toLowerCase();
    if (["w","a","s","d"].includes(key)) {
        fetch("/input", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({key: key})
        });
    }
});
</script>

</body>
</html>
"""

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
</head>
<body style="text-align:center;margin-top:50px;">
    <h2>Enter Password</h2>
    <form method="POST">
        <input type="password" name="password" />
        <button type="submit">Login</button>
    </form>
    <p style="color:red;">{{error}}</p>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("control"))
        else:
            return render_template_string(LOGIN_PAGE, error="Wrong password")
    return render_template_string(LOGIN_PAGE, error="")

@app.route("/control")
def control():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template_string(HTML_PAGE)

@app.route("/input", methods=["POST"])
def handle_input():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 403

    data = request.get_json()
    key = data.get("key")

    logging.info(f"Received key: {key}")

    # 👉 here you can add your own logic
    # e.g. control robot, send command to game, etc.

    return jsonify({"status": "ok", "key": key})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6666)