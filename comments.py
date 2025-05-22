from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/', methods=['POST'])
def comment():
    if request.method == 'POST':
        data = request.get_json()
        if data and 'text' in data:
            try:
                # Step 1: Translate the text to English using deep-translator
                translator = GoogleTranslator(source='auto', target='en')
                translation = translator.translate(data['text'])
                print('Translation:', translation)

                # Step 2: Load the dataset
                df = pd.read_csv("commentsdata.csv")  # Make sure to have a CSV with 'text' and 'label' columns
                X = df['text']
                y = df['label']

                # Step 3: Use TF-IDF Vectorizer to convert text to numerical features
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
                X_features = vectorizer.fit_transform(X)

                # Step 4: Apply K-Means Clustering
                num_clusters = 2  # We assume two clusters: abusive and non-abusive
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(X_features)

                # Step 5: Predict the cluster of new text
                new_text = [translation]
                new_text_features = vectorizer.transform(new_text)
                prediction = kmeans.predict(new_text_features)
                print("Predicted cluster:", prediction)

                # Step 6: Map clusters to labels (0: Non-abusive, 1: Abusive)
                abusive_cluster = 1 if sum(kmeans.labels_) > len(kmeans.labels_) // 2 else 0
                predicted_label = abusive_cluster if prediction[0] == abusive_cluster else 1 - abusive_cluster

                if predicted_label == 1:
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
