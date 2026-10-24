"""Unigram, bigram, and trigram vocab helpers used by the bag-of-words model."""

import nltk


def find_universal_ngrams(documents, uni_thres, bi_thres, tri_thres):
    """Keep n-grams that appear often enough in the full document collection."""
    words = " ".join(documents.tolist()).split(" ")
    word_list = [document.split(" ") for document in documents]

    unigrams = []
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    for word, freq in word_freq.items():
        if freq >= uni_thres:
            unigrams.append(word)

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_documents(word_list)
    bigrams = []
    bi_freq_dict = dict(bigram_finder.ngram_fd)
    for bigram in bigram_finder.above_score(bigram_measures.pmi, 0):
        if bi_freq_dict[bigram] >= bi_thres:
            bigrams.append(bigram)

    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_documents(word_list)
    trigrams = []
    tri_freq_dict = dict(trigram_finder.ngram_fd)
    for trigram in trigram_finder.above_score(trigram_measures.pmi, 0):
        if tri_freq_dict[trigram] >= tri_thres:
            trigrams.append(trigram)
    return unigrams, bigrams, trigrams


def find_ngrams(words, unigrams, bigrams, trigrams):
    """Return the n-grams from one token list that sit in the shared vocab."""
    uni = [word for word in words if word in unigrams]
    bi = [" ".join(pair) for pair in zip(words[:-1], words[1:]) if pair in bigrams]
    tri = [
        " ".join(triple)
        for triple in zip(words[:-2], words[1:-1], words[2:])
        if triple in trigrams
    ]
    return uni + bi + tri
