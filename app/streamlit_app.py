import streamlit as st
import requests
import os

st.title("🎬 IMDB Movie Review Sentiment Analysis")

# Backend URL (local for dev, remote for deploy)
API_URL = os.getenv("API_URL", "https://movie-sentiment-analysis-bse6.onrender.com/predict")


review = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    try:
        response = requests.post(API_URL, json={"review": review})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Sentiment: **{result['sentiment']}**")
        else:
            st.error("⚠️ Error: Could not get response from API")
    except Exception as e:
        st.error(f"⚠️ Connection error: {e}")
