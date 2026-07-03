from transformers import AutoTokenizer, TFBartForConditionalGeneration, pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFBartForConditionalGeneration.from_pretrained(model_name, use_safetensors=False)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_reviews(text):
    summarizer = load_summarizer()
    # Limit input to the first 600 words to ensure it fits within the model's 1024 token limit
    truncated_text = " ".join(text.split()[:600])
    result = summarizer(truncated_text, max_length=130, min_length=30, do_sample=False)
    return result[0]["summary_text"]
