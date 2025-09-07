import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Load test dataset
df = pd.read_csv("data/IMDB Dataset.csv").sample(500)  # small sample for quick evaluation
df["clean_review"] = df["review"].apply(clean_text)

X = tfidf.transform(df["clean_review"])
y = df["sentiment"]

# Predict
y_pred = model.predict(X)

print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
