from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Server is running", 200

@app.route('/', methods=['POST'])
def comment():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        if data and 'text' in data:
            try:
                translator = GoogleTranslator(source='auto', target='en')
                translation = translator.translate(data['text'])
                print('translation:', translation)

                # Load the dataset
                df = pd.read_csv("commentsdata.csv")

                # Prepare the features and target variable
                X = df['text']
                y = df['label']

                # Convert text to numerical features using TF-IDF
                vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_features=1000)
                X_features = vectorizer.fit_transform(X)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_features, y, test_size=0.3, random_state=42)

                # Train the Logistic Regression model
                model = LogisticRegression(C=0.001, penalty='l2', class_weight='balanced', solver='liblinear')
                model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy: {:.2f}%".format(accuracy * 100))

                # Predict whether new text is abusive
                new_text = [translation]
                new_text_features = vectorizer.transform(new_text)
                prediction = model.predict(new_text_features)
                print("Predicted label:", prediction)
                if prediction[0] == 1:
                    print("Prediction for new text: Abusive")
                    return jsonify({"message": "Abusive"}), 200
                else:
                    print("Prediction for new text: Not Abusive")
                    return jsonify({"message": "Not Abusive"}), 200
            except Exception as e:
                return jsonify({'error': f"An error occurred: {str(e)}"}), 500
        else:
            return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run()
    print("Server Listening....")

