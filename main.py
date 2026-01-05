import os
from src.preprocess import preprocess
from src.sentiment import analyze_sentiment
from src.topics import topic_modeling
from src.summarize import summarize_reviews

RAW_DATA = "data/raw/reviews.csv"
PROCESSED_DATA = "data/processed/cleaned_reviews.csv"

def run_pipeline():
    if not os.path.exists(RAW_DATA):
        raise FileNotFoundError(
            "Dataset not found. Place reviews.csv inside data/raw/"
        )

    os.makedirs("data/processed", exist_ok=True)

    df = preprocess(RAW_DATA, PROCESSED_DATA)

    df["sentiment"] = None

    sample_size = 200
    df.loc[:sample_size-1, "sentiment"] = analyze_sentiment(
        df.loc[:sample_size-1, "cleaned_review"].tolist()
    )

    topics = topic_modeling(
        df["cleaned_review"].dropna().tolist()[:1000]
    )

    summary = summarize_reviews(
        " ".join(df["cleaned_review"].tolist()[:20])
    )

    df.to_csv(PROCESSED_DATA, index=False)

    with open("data/processed/topics.txt", "w", encoding="utf-8") as f:
        for t in topics:
            f.write(str(t) + "\n")

    with open("data/processed/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()