from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset and train model once when app starts
df = pd.read_csv("commentsdata.csv")

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_features = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.3, random_state=42)

model = LogisticRegression(C=0.001, penalty='l2', class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train)

# Optional: print accuracy to logs
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

@app.route('/', methods=['GET'])
def home():
    return "Server is running", 200

@app.route('/', methods=['POST'])
def comment():
    data = request.get_json()
    print("Received data:", data)
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    try:
        translator = GoogleTranslator(source='auto', target='en')
        translation = translator.translate(data['text'])
        print('Translation:', translation)

        new_text_features = vectorizer.transform([translation])
        prediction = model.predict(new_text_features)
        print("Predicted label:", prediction)

        message = "Abusive" if prediction[0] == 1 else "Not Abusive"
        return jsonify({"message": message}), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run()
    print("Server Listening....")

