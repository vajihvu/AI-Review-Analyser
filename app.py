# Mock torch to prevent Hugging Face transformers from raising NameError when PyTorch is not installed
import builtins
class DummyTorch:
    pass
builtins.torch = DummyTorch()

import streamlit as st
import pandas as pd
import re
from src.preprocess import clean_text
from src.sentiment import analyze_sentiment
from src.topics import topic_modeling
from src.summarize import summarize_reviews

st.set_page_config(page_title="AI Customer Review Analyzer", layout="wide")
st.title("AI Customer Review Analyzer")

# --- Sidebar: Custom Review Sentiment Analyzer ---
st.sidebar.title("🔍 Quick Sentiment Check")
st.sidebar.write("Type or paste a custom review below to immediately check if it is positive or negative.")
custom_review = st.sidebar.text_area("Custom Review Text", placeholder="Write your review here...", height=150)
if st.sidebar.button("Analyze Custom Sentiment", use_container_width=True):
    if custom_review.strip():
        with st.sidebar.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment([custom_review])[0]
            if sentiment == "POSITIVE":
                st.sidebar.success(f"### Result: **POSITIVE** 👍")
            else:
                st.sidebar.error(f"### Result: **NEGATIVE** 👎")
    else:
        st.sidebar.warning("Please enter some text first.")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_reviews.csv")

df = load_data()

# Initialize sample in session state so it remains stable during other page actions
if "sample_df" not in st.session_state:
    st.session_state.sample_df = df.sample(10)

st.subheader("Sample Reviews")
st.dataframe(st.session_state.sample_df)

if st.button("🔄 Shuffle Samples"):
    st.session_state.sample_df = df.sample(10)
    st.rerun()

if st.button("Run Analysis"):

    sample_df = st.session_state.sample_df.copy()

    with st.spinner("Running sentiment analysis..."):
        sample_df["sentiment"] = analyze_sentiment(
            sample_df["cleaned_review"].tolist()
        )
        st.success("Sentiment analysis completed")

    st.subheader("Sentiment Results")
    st.dataframe(
        sample_df[["cleaned_review", "sentiment"]]
    )

    with st.spinner("Extracting topics..."):
        # We extract topics from a larger slice of the dataset so that the topics are meaningful/stable
        topics = topic_modeling(
            df["cleaned_review"].dropna().tolist()[:500]
        )
        st.subheader("Key Topics Extracted (from dataset)")
        for idx, topic_str in topics:
            words = re.findall(r'"([^"]+)"', topic_str)
            words = [w for w in words if w != "br"][:5]
            st.markdown(f"**Topic #{idx + 1}:** " + " ".join([f"`{w}`" for w in words]))

    with st.spinner("Generating summary..."):
        summary = summarize_reviews(
            " ".join(sample_df["review_text"].tolist())
        )
        st.subheader("Abstractive Review Summary (for current samples)")
        st.write(summary)
