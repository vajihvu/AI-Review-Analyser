from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", framework="tf", dtype=None)

def analyze_sentiment(texts):
    model = load_sentiment_model()
    return [r["label"] for r in model(texts, truncation=True)]
