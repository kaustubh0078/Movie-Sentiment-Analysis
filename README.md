# 🎬 IMDB Movie Review Sentiment Analysis

This project is an end-to-end **Sentiment Analysis system** built using the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) of movie reviews.
The goal is to classify each review as **Positive** or **Negative** using a machine learning model.

The system includes:
- Data preprocessing (cleaning, lemmatization, stopword removal)
- Feature extraction with TF-IDF
- Logistic Regression classifier
- REST API built with FastAPI
- Interactive frontend built with Streamlit
- Docker support for easy deployment
- Live deployment on **Render (backend)** and **Streamlit Cloud (frontend)**

---

## ⚙️ Setup & Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/](https://github.com/)<kaustubh0078>/imdb-sentiment-analysis.git
    cd imdb-sentiment-analysis
    ```

2.  **Create and activate virtual environment**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the model**
    ```bash
    python src/train.py
    ```
    This generates the following files inside `models/`:
    * `sentiment_model.pkl`
    * `tfidf_vectorizer.pkl`

---

## 🚀 Running Locally

1.  **Start the FastAPI backend:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at: 👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

2.  **Run the Streamlit frontend (in another terminal):**
    ```bash
    streamlit run app/streamlit_app.py
    ```
    The UI will open at: 👉 [http://localhost:8501](http://localhost:8501)

---

## 🐳 Running with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t imdb-sentiment .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 imdb-sentiment
    ```
    The FastAPI backend will be accessible at [http://localhost:8000](http://localhost:8000).

---

## ☁️ Deployment

### Backend (FastAPI)

Deployed on **Render**. Example link:
[https://movie-sentiment-analysis-bse6.onrender.com/docs](https://movie-sentiment-analysis-bse6.onrender.com/docs)

### Frontend (Streamlit)

Deployed on **Streamlit Cloud**. Example link:
[https://your-username-imdb-sentiment.streamlit.app](https://movie-sentiment-analysis-bkwzffwa5bh9zlljydrebn.streamlit.app/)

When deploying the Streamlit app, you must set the `API_URL` environment variable to point to your deployed backend service. For example:
API_URL=https://movie-sentiment-analysis-bse6.onrender.com/predict


---

## 🧪 Example Predictions

| Review                                         | Prediction                                     |
| ---------------------------------------------- | ---------------------------------------------- |
| "This movie was absolutely fantastic!"         | Positive                                       |
| "The worst film I’ve ever seen."               | Negative                                       |
| "It’s so bad that it’s actually entertaining." | Negative (sarcasm is difficult for this model) |

---

## 🔮 Future Work

- [ ] Extend TF-IDF with bigrams/trigrams to capture more context.
- [ ] Experiment with deep learning models like LSTMs or Transformers (e.g., BERT).
- [ ] Improve handling of sarcasm and mixed sentiments.
- [ ] Add a database and user authentication for storing reviews.
- [ ] Deploy both backend and frontend in a single container for simplicity.

---

## 📜 License

This project is licensed under the MIT License © 2025.

---

### ✨ Author

Developed by [Kaustubh Jaiswal](https://github.com/<kaustubh0078>)
