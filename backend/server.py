"""Privacy-first gait-motion research demo. No clinical predictions or server-side uploads."""
from flask import Flask, jsonify, render_template
app = Flask(__name__, template_folder="frontend/templates")

@app.get("/")
@app.get("/welcome")
@app.get("/index")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify(status="ok", application="gait-motion-demo", inference="not_available")

@app.post("/upload")
def upload():
    return jsonify(error="Videos are processed locally in your browser; no uploads are accepted."), 410

@app.post("/predict")
def predict():
    return jsonify(error="Clinical predictions are unavailable. This research demonstration measures motion only."), 501

if __name__ == "__main__":
    app.run(debug=False)
