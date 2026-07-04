---
title: AI Customer Review Analyzer
emoji: 📊
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
---

# AI Customer Review Analyzer 📊

An end-to-end NLP system for analyzing customer reviews. It performs text preprocessing, sentiment analysis, topic modeling, and abstractive summarization. 

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/vajihvu/nlp-insights)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/vajihvu/AI-Review-Analyser)

---

## 🚀 Live Demo
Access the live Streamlit dashboard on Hugging Face Spaces:  
👉 **[Live App Link](https://huggingface.co/spaces/vajihvu/nlp-insights)**

---

## ✨ Key Features
- **Data Preprocessing:** Standardized pipeline to clean HTML tags, punctuation, and default NLTK stopwords.
- **TF-Powered Sentiment Analysis:** Sentiment classification using a pre-trained DistilBERT model running on the **TensorFlow** backend.
- **Custom Stopword Topic Modeling:** Extract distinct, non-overlapping keywords using GenSim LDA with domain-specific stopword filtering.
- **Abstractive Summarization:** Auto-generate paragraph summaries from a set of reviews using BART-Large-CNN on **TensorFlow**.
- **Interactive Dashboard:** 
  - **🔍 Quick Sentiment Check:** Input a custom review in the sidebar to check if it's positive or negative.
  - **🔄 Shuffle Samples:** Dynamically randomize reviews on screen to run sentiment analysis and summarization on new subsets.

---

## 📁 Repository Structure
*   `app.py` - The Streamlit interactive web application.
*   `main.py` - Local command-line execution script for the NLP pipeline.
*   `Review_Analysis_Pipeline.ipynb` - Clean, execution-only Jupyter Notebook walking through the pipeline steps.
*   `src/` - Core module scripts:
    *   `src/__init__.py` - Package entry point containing the PyTorch compatibility mock.
    *   `src/preprocess.py` - Text cleaning utilities.
    *   `src/sentiment.py` - Sentiment analyzer loader (TensorFlow).
    *   `src/topics.py` - LDA Topic modeling with domain-specific stopwords.
    *   `src/summarize.py` - Abstractive BART summarizer (TensorFlow).
*   `data/` - Dataset directories (raw & processed).

---

## ⚙️ How to Run Locally

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/vajihvu/AI-Review-Analyser.git
cd AI-Review-Analyser
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Script
Executes the data processing, runs sentiment analysis, topic extraction, and summaries, and outputs generated files in `data/processed/`.
```bash
python main.py
```

### 4. Run Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack
- **Language:** Python 3.13
- **Deep Learning Framework:** TensorFlow & Keras (compatibility layer: `tf-keras`)
- **NLP Models:** Hugging Face Transformers (`DistilBERT`, `BART-Large-CNN`), NLTK, GenSim (LDA)
- **Data Handling:** Pandas, NumPy
- **Web UI:** Streamlit

---

## 📊 Dataset Reference
Uses the **Amazon Fine Food Reviews** dataset (Kaggle), filtering down to:
- `Text` – Cleaned review content.
- `Score` – Ratings (1-5).
- `Time` – Timestamp of the review.
