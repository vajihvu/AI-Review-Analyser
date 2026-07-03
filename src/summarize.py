from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="tf", dtype=None)

def summarize_reviews(text):
    summarizer = load_summarizer()
    # Limit input to the first 600 words to ensure it fits within the model's 1024 token limit
    truncated_text = " ".join(text.split()[:600])
    result = summarizer(truncated_text, max_length=130, min_length=30, do_sample=False)
    return result[0]["summary_text"]
