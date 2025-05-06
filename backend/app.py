from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Load datasets
fake_df = pd.read_csv(os.path.join(DATASET_DIR, 'fake_labeled.csv'))
real_df = pd.read_csv(os.path.join(DATASET_DIR, 'real_labeled.csv'))

# Combine datasets
fake_df['label'] = 1
real_df['label'] = 0
df = pd.concat([fake_df, real_df])

# Preprocess text
df['text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as model_file:
    pickle.dump(model, model_file)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Load model and vectorizer
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as model_file:
    model = pickle.load(model_file)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detector API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "Invalid request! 'text' key is required."}), 400

        news_text = data["text"]
        cleaned_text = clean_text(news_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        result = "Real News" if prediction == 0 else "Fake News"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)