from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "commentsdata.csv"

# ==============================
# TRAIN / LOAD MODEL
# ==============================

def train_and_save_model():
    print("Training model...")

    df = pd.read_csv(DATA_PATH)

    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_features = vectorizer.fit_transform(X)

    model = LogisticRegression(
        C=0.001,
        penalty='l2',
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_features, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model trained & saved")
    return model, vectorizer


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Loading saved model...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        model, vectorizer = train_and_save_model()

    return model, vectorizer


model, vectorizer = load_model()

# ==============================
# SAVE DATA
# ==============================

def save_to_dataset(text, label):
    try:
        new_data = pd.DataFrame({
            "text": [text],
            "label": [label]
        })

        new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
        print("Saved to dataset")

    except Exception as e:
        print("Error saving:", str(e))


# ==============================
# ROUTES
# ==============================

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server is running"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    try:
        input_text = data['text']

        # Translate to English
        translator = GoogleTranslator(source='auto', target='en')
        translation = translator.translate(input_text)

        # Predict
        features = vectorizer.transform([translation])
        prediction = int(model.predict(features)[0])

        result = "Abusive" if prediction == 1 else "Not Abusive"

        # ✅ Append to dataset
        save_to_dataset(translation, prediction)

        return jsonify({
            "original": input_text,
            "translated": translation,
            "prediction": result
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================
# MANUAL RETRAIN ROUTE
# ==============================

@app.route('/retrain', methods=['POST'])
def retrain():
    global model, vectorizer
    model, vectorizer = train_and_save_model()
    return jsonify({"message": "Model retrained successfully"}), 200


# ==============================
# MAIN
# ==============================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 1000))
    app.run(host='0.0.0.0', port=port)