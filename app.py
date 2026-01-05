import streamlit as st
import pandas as pd
from src.sentiment import analyze_sentiment
from src.topics import topic_modeling
from src.summarize import summarize_reviews

st.set_page_config(page_title="AI Customer Review Analyzer", layout="wide")
st.title("AI Customer Review Analyzer")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_reviews.csv")

df = load_data()

st.subheader("Sample Reviews")
st.dataframe(df.head(10))

if st.button("Run Analysis"):

    # Initialize sentiment column
    df["sentiment"] = None

    with st.spinner("Running sentiment analysis..."):
        df.loc[:29, "sentiment"] = analyze_sentiment(
            df.loc[:29, "cleaned_review"].tolist()
        )
        st.success("Sentiment analysis completed")

    st.subheader("Sentiment Results")
    st.dataframe(
        df.loc[:29, ["cleaned_review", "sentiment"]]
    )

    with st.spinner("Extracting topics..."):
        topics = topic_modeling(
            df["cleaned_review"].dropna().tolist()[:500]
        )
        for t in topics:
            st.write(t)

    with st.spinner("Generating summary..."):
        summary = summarize_reviews(
            " ".join(df["cleaned_review"].tolist()[:20])
        )
        st.write(summary)
