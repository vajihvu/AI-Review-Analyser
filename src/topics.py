from gensim import corpora, models

def topic_modeling(texts, num_topics=5):
    tokens = [t.split() for t in texts]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    lda = models.LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42
    )
    return lda.print_topics()
