# 🛡️ AI-Based Fake News Detection Platform

A complete, self-contained full-stack project that uses a machine learning
model (TF‑IDF + Logistic Regression) to classify a piece of news text as
**REAL** or **FAKE**, with a Flask REST API backend and a clean HTML/CSS/JS
frontend.

```
fake-news-detector/
├── backend/
│   ├── app.py                  # Flask API server (serves frontend + /api/predict)
│   ├── requirements.txt        # Python dependencies
│   ├── data/
│   │   ├── generate_dataset.py # Builds the training dataset (dataset.csv)
│   │   └── dataset.csv         # Generated labeled dataset (656 rows)
│   └── model/
│       ├── train_model.py      # Trains TF-IDF + Logistic Regression pipeline
│       └── fake_news_pipeline.joblib  # Trained, saved model (pre-built)
├── frontend/
│   ├── index.html              # UI
│   ├── style.css               # Styling (dark theme)
│   └── script.js               # Calls the backend API, renders result
└── README.md
```

## How it works (architecture)

1. **Dataset** (`backend/data/generate_dataset.py`): produces a labeled
   CSV of `text,label` pairs (`label`: `0` = REAL, `1` = FAKE). REAL samples
   are written in a measured, attributed journalistic style; FAKE samples
   mimic common misinformation patterns (sensational claims, excessive
   punctuation, urgency, lack of sourcing).
2. **Model training** (`backend/model/train_model.py`): cleans the text
   (lowercase, strip URLs/punctuation/numbers), vectorizes it with
   **TF‑IDF** (unigrams + bigrams, 8,000 features), and trains a
   **Logistic Regression** classifier. The fitted pipeline (vectorizer +
   classifier bundled together) is saved as `fake_news_pipeline.joblib`.
3. **Backend API** (`backend/app.py`): loads the saved pipeline once at
   startup and exposes `POST /api/predict`, which takes `{"text": "..."}`
   and returns the predicted label, confidence, class probabilities, and a
   few human-readable "signals" (heuristic red flags like excessive
   capitalization or missing attribution) to make the result explainable.
   It also serves the frontend at `/`.
4. **Frontend** (`frontend/`): a single page where a user pastes text,
   clicks **Analyze**, and sees a REAL/FAKE badge, a confidence bar, and
   the list of signals noticed — all styled with a dark, modern theme.

## ⚠️ Important note on the dataset

This build environment has no internet access to download a public
labeled dataset (e.g. Kaggle's ISOT Fake/Real News or LIAR dataset), so
`generate_dataset.py` synthesizes a **demo** dataset that captures the
*stylistic* patterns typically associated with misinformation (sensational
language, urgency, lack of sourcing) versus straight factual reporting.
This is enough to build and demonstrate a fully working pipeline end to
end, but it will **not** generalize as well as a model trained on a real,
human-labeled corpus of tens of thousands of actual news articles.

**To make this production-grade:** download a real dataset (ISOT
Fake-and-Real-News, LIAR, or FakeNewsNet), drop it in as
`backend/data/dataset.csv` with the same `text,label` columns, and rerun
`train_model.py`. No other code changes are needed.

## Setup & running locally

Requires Python 3.9+.

```bash
cd fake-news-detector/backend
pip install -r requirements.txt

# (Optional — a trained model is already included in model/fake_news_pipeline.joblib)
python data/generate_dataset.py     # regenerate dataset.csv
python model/train_model.py         # retrain and save the model

# Start the server
python app.py
```

Then open **http://localhost:5000** in your browser. The Flask app serves
both the API and the frontend from the same origin, so there's no CORS
setup needed for local use (flask-cors is included anyway in case you
split the frontend onto a different host/port later).

## API reference

### `POST /api/predict`

**Request body:**
```json
{ "text": "Some news headline or article text..." }
```

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 90.6,
  "probabilities": { "real": 9.4, "fake": 90.6 },
  "flags": [
    "Uses multiple sensational / clickbait phrases",
    "Excessive exclamation marks",
    "No attribution to a named source, official, or study"
  ]
}
```

### `GET /health`
Returns `{"status": "ok"}` — useful for uptime checks / load balancers.

## Deploying

- **Backend**: any WSGI host (Render, Railway, Fly.io, a VM with gunicorn +
  nginx). Run with `gunicorn -w 2 -b 0.0.0.0:$PORT app:app` in production
  instead of the Flask dev server.
- **Frontend**: already served by Flask, so deploying the backend deploys
  the whole app. If you prefer a separate static host (Netlify/Vercel) for
  the frontend, just update `API_URL` in `frontend/script.js` to point at
  your backend's URL and enable CORS (already included).

## Extending this project

- Swap in a **transformer model** (e.g. a fine-tuned DistilBERT) for
  higher accuracy on real-world text — replace the pipeline in
  `train_model.py` and adjust `app.py`'s `predict()` accordingly.
- Add a **browser extension** or bookmarklet that sends the visible page
  text to `/api/predict`.
- Add **source credibility checks** (domain reputation, publish date,
  cross-referencing claims via a news API) alongside the ML score.
- Persist submitted articles + verdicts to a database for building an
  audit trail or a feedback loop for retraining.
- Add user auth + a history dashboard if this becomes a multi-user
  platform rather than a single-page tool.
