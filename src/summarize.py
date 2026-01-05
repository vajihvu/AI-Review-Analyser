from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_reviews(text):
    summarizer = load_summarizer()
    result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return result[0]["summary_text"]
