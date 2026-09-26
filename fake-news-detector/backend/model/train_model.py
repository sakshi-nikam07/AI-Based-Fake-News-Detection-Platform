"""
train_model.py
---------------
Trains a TF-IDF + Logistic Regression text classifier to distinguish
FAKE (1) vs REAL (0) news text, and saves the trained pipeline to disk
so the Flask API can load it instantly at request time (no retraining
per request).

Run: python train_model.py
Produces: fake_news_pipeline.joblib   (vectorizer + classifier, bundled)
"""

import os
import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_news_pipeline.joblib")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, preds))

    # Note: we only pickle the sklearn pipeline itself (not the clean_text
    # function) so the saved file has no dependency on this module's path.
    # app.py defines an identical clean_text() to preprocess input the same way.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
