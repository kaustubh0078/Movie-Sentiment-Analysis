from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.preprocess import clean_text

app = FastAPI()

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

class ReviewInput(BaseModel):
    review: str

@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    clean = clean_text(data.review)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]
    return {"sentiment": prediction}
