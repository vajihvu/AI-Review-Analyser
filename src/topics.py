from gensim import corpora, models

# Generic review filler words that dominate all topics mathematically
GENERIC_STOPWORDS = {
    "like", "good", "great", "taste", "product", "one", "get", "would", 
    "find", "really", "love", "buy", "try", "use", "make", "flavor", 
    "eat", "food", "order", "box", "bag", "pack", "best", "much", "br",
    "dont", "even", "time", "little", "also", "well", "better", "could"
}

def topic_modeling(texts, num_topics=5):
    tokens = [[w for w in t.split() if w not in GENERIC_STOPWORDS] for t in texts]
    
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    
    corpus = [dictionary.doc2bow(t) for t in tokens]
    lda = models.LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=15,
        random_state=42
    )
    return lda.print_topics()
