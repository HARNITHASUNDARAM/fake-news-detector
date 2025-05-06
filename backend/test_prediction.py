import pickle
import os
from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_news(news_text):
    """Predicts whether news is fake or real."""
    cleaned_text = clean_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return "Fake News" if prediction == 1 else "Real News"

if __name__ == "__main__":
    news_article = input("Enter a news article: ")
    result = predict_news(news_article)
    print("\nPrediction:", result)