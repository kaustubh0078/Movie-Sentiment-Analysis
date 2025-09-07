import joblib
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

def predict_sentiment(review: str) -> str:
    clean = clean_text(review)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]
    return prediction
