import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Preprocess text
df["clean_review"] = df["review"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=400, solver="saga")
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
