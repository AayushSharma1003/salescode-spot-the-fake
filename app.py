"""
Tiny web demo for the real-vs-screen detector.
GET /         -> upload page
POST /predict -> {"score": float, "latency_ms": float}
"""

import io
import time
import tempfile
import warnings
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from predict import predict, _load

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB cap

# Warm the model on boot so the first request isn't slow.
_load()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict_route():
    if "image" not in request.files:
        return jsonify(error="no image uploaded"), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify(error="empty filename"), 400

    # features.py expects a path, so spool the upload to a temp file.
    suffix = Path(f.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        f.save(tmp.name)
        t0 = time.perf_counter()
        try:
            score = predict(tmp.name)
        except Exception as e:
            return jsonify(error=f"prediction failed: {e}"), 500
        latency_ms = (time.perf_counter() - t0) * 1000

    return jsonify(score=score, latency_ms=round(latency_ms, 1))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="image too large (15 MB max)"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)