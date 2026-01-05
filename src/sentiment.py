from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

def analyze_sentiment(texts):
    model = load_sentiment_model()
    return [r["label"] for r in model(texts, truncation=True)]
