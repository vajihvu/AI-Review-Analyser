from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=False)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(texts):
    model = load_sentiment_model()
    return [r["label"] for r in model(texts, truncation=True)]
