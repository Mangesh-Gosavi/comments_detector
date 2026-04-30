import os
import gc
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "commentsdata.csv"

MAX_TRAIN_ROWS = 3000
MAX_FEATURES = 5000
CONFIDENCE_THRESHOLD = 0.9


def train_and_save_model():
    print("Training model (fallback)...")

    df = pd.read_csv(DATA_PATH, usecols=["text", "label"])

    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(n=MAX_TRAIN_ROWS, random_state=42)

    df = df.dropna(subset=["text", "label"])

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        min_df=2,
        sublinear_tf=True,
        dtype="float32",
    )
    X_features = vectorizer.fit_transform(X)

    model = LogisticRegression(
        C=0.1,
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=200,
    )
    model.fit(X_features, y)

    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)

    del df, X, y, X_features
    gc.collect()

    print("Model trained & saved")
    return model, vectorizer


def load_or_train():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            print("Loading saved model...")
            return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    except Exception as e:
        print(f"Failed to load model files, retraining: {e}")

    return train_and_save_model()


model, vectorizer = load_or_train()


def save_to_dataset(text, label):
    try:
        pd.DataFrame({"text": [text], "label": [label]}).to_csv(
            DATA_PATH, mode="a", header=False, index=False
        )
    except Exception as e:
        print("Error saving:", str(e))


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        input_text = data["text"]

        translation = GoogleTranslator(source="auto", target="en").translate(input_text)

        features = vectorizer.transform([translation])
        probabilities = model.predict_proba(features)[0]
        prediction = int(probabilities.argmax())
        confidence = float(probabilities.max())

        result = "Abusive" if prediction == 1 else "Not Abusive"

        if confidence >= CONFIDENCE_THRESHOLD:
            save_to_dataset(translation, prediction)

        return jsonify({
            "original": input_text,
            "translated": translation,
            "prediction": result,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    global model, vectorizer
    model, vectorizer = train_and_save_model()
    return jsonify({"message": "Model retrained successfully"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port)
