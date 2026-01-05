import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df[["Text", "Score", "Time"]]
    df.rename(columns={"Text": "review_text", "Score": "rating"}, inplace=True)
    df["cleaned_review"] = df["review_text"].astype(str).apply(clean_text)
    df.to_csv(output_path, index=False)
    return df
