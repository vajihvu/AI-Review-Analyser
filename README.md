# AI Customer Review Analyzer

End-to-end NLP system for analyzing customer reviews using sentiment analysis,
topic modeling, and abstractive summarization.

## Features
- Data preprocessing and cleaning
- Transformer-based sentiment analysis
- Topic modeling with LDA
- Abstractive summarization
- Interactive Streamlit dashboard

## How to Run
pip install -r requirements.txt

python main.py

streamlit run app.py

## Tech Stack
- Language: Python
- NLP: Hugging Face Transformers, NLTK, Gensim
- Data Handling: Pandas, NumPy
- Visualization: Streamlit
- Model Backend: PyTorch
- Development: VS Code, Git, GitHub

## Dataset
Amazon Fine Food Reviews Dataset

Source: Kaggle

Only the following columns are used:
- Text – Review text
- Score – Rating (1–5)
- Time – Review timestamp
  
## Output
- Sentiment classification (Positive / Negative)
- Dominant review topics
- Executive-level text summary
- Interactive UI for inspection
