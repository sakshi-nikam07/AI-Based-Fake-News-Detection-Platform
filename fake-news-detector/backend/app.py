"""
app.py
------
Flask backend for the AI-Based Fake News Detection Platform.

Endpoints:
  GET  /                 -> serves the frontend (index.html)
  GET  /health           -> simple health check
  POST /api/predict      -> {"text": "..."} -> prediction JSON

Run:
  python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import re
import string
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_pipeline.joblib")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Load model once at startup (not per-request) for speed.
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model file not found. Run `python model/train_model.py` first "
                "(after generating the dataset with `python data/generate_dataset.py`)."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def clean_text(text: str) -> str:
    """Must mirror the preprocessing used in train_model.py exactly."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Lightweight, explainable heuristic signals shown alongside the ML verdict.
# These do NOT affect the model's prediction; they're surfaced in the UI to
# help users understand *why* something might look suspicious.
# ---------------------------------------------------------------------------
SENSATIONAL_WORDS = [
    "shocking", "breaking", "exposed", "secret", "hidden", "miracle",
    "wake up", "won't believe", "they don't want you to know",
    "leaked", "banned", "cure", "conspiracy", "hoax", "cover up", "coverup",
]


def heuristic_flags(raw_text: str):
    lower = raw_text.lower()
    flags = []
    if sum(1 for w in SENSATIONAL_WORDS if w in lower) >= 2:
        flags.append("Uses multiple sensational / clickbait phrases")
    if raw_text.count("!") >= 3:
        flags.append("Excessive exclamation marks")
    letters = [c for c in raw_text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.3:
        flags.append("Unusually high proportion of capital letters")
    if re.search(r"share this|forward this|before it('s| is) (deleted|too late)", lower):
        flags.append("Urges urgent sharing/forwarding, a common misinformation tactic")
    if not re.search(r"\b(said|reported|according to|confirmed|announced|study|data|official)\b", lower):
        flags.append("No attribution to a named source, official, or study")
    return flags


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide non-empty 'text' in the request body."}), 400
    if len(text) < 15:
        return jsonify({"error": "Please provide a longer piece of text (at least ~15 characters)."}), 400

    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    cleaned = clean_text(text)
    proba = pipeline.predict_proba([cleaned])[0]  # [P(real), P(fake)]
    pred_label = int(pipeline.predict([cleaned])[0])  # 0 = real, 1 = fake

    confidence = float(proba[pred_label])
    result = {
        "label": "FAKE" if pred_label == 1 else "REAL",
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "real": round(float(proba[0]) * 100, 2),
            "fake": round(float(proba[1]) * 100, 2),
        },
        "flags": heuristic_flags(text),
    }
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
