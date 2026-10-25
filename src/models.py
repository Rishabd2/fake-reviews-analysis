"""Detectors compared in this repo. Each takes train/test frames and returns predictions.

Frames carry `text`, `polarity`, `y`, plus lexicon columns (see lexicon.py) and,
for the Random Forest baseline, the preprocessed columns from data_prep.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import lexicon
import ngram

SEED = 0
TOKEN_PATTERN = r"(?u)\b\w+\b"  # keeps the one-letter pronoun "i"
PRONOUNS = sorted(set().union(*(lexicon.WORDS[g] for g in lexicon.GROUPS["pronouns"])))
RF_LING = ["num_word", "num_coreword", "num_stopword", "num_char", "char_per_word", "num_first_sing"]


def _inner_cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def _tfidf(stop_words=None) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        token_pattern=TOKEN_PATTERN, stop_words=stop_words,
    )


def _lex_matrix(train, test, cols):
    scaler = StandardScaler().fit(train[cols].to_numpy(float))
    return (
        sparse.csr_matrix(scaler.transform(train[cols].to_numpy(float))),
        sparse.csr_matrix(scaler.transform(test[cols].to_numpy(float))),
    )


def _fit_linear(X_tr, y_tr, X_te, kind: str):
    if kind == "lr":
        est = LogisticRegression(max_iter=5000, random_state=SEED)
    else:
        est = LinearSVC(random_state=SEED, max_iter=20000)
    grid = GridSearchCV(est, {"C": [0.1, 1, 10, 100]}, cv=_inner_cv(), scoring="f1_macro")
    grid.fit(X_tr, y_tr)
    return grid.predict(X_te)


def linear_model(kind: str = "lr", ngrams: bool = True, lex_cols=None,
                 drop_pronoun_tokens: bool = False) -> Callable:
    """TF-IDF word 1-2 grams and/or standardized lexicon features into LR or linear SVM."""

    def run(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        blocks_tr, blocks_te = [], []
        if ngrams:
            vec = _tfidf(stop_words=PRONOUNS if drop_pronoun_tokens else None)
            blocks_tr.append(vec.fit_transform(train["text"]))
            blocks_te.append(vec.transform(test["text"]))
        if lex_cols:
            a, b = _lex_matrix(train, test, list(lex_cols))
            blocks_tr.append(a)
            blocks_te.append(b)
        X_tr = sparse.hstack(blocks_tr).tocsr()
        X_te = sparse.hstack(blocks_te).tocsr()
        return _fit_linear(X_tr, train["y"].to_numpy(), X_te, kind)

    return run


def rf_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The saved notebook's Random Forest: PMI-filtered 1-3 gram counts, rating, six length/pronoun counts.

    Vocabulary thresholds (16, 16, 10) and forest settings are the notebook's grid-search result.
    The vocabulary is built from the training fold only.
    """
    uni, bi, tri = ngram.find_universal_ngrams(train["PP_Review"], 16, 16, 10)
    vocab = uni + [" ".join(b) for b in bi] + [" ".join(t) for t in tri]
    index = {w: i for i, w in enumerate(vocab)}
    uni_s, bi_s, tri_s = set(uni), set(bi), set(tri)

    def matrix(df):
        rows, cols = [], []
        for r, words in enumerate(df["Word_List"]):
            for w in ngram.find_ngrams(words, uni_s, bi_s, tri_s):
                rows.append(r)
                cols.append(index[w])
        X = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(vocab))).toarray()
        rating = (df["polarity"] == "positive").map({True: 5, False: 1}).to_numpy()[:, None]
        return np.hstack([X, rating, df[RF_LING].to_numpy(float)])

    rf = RandomForestClassifier(
        criterion="gini", max_depth=20, max_features=5, max_samples=None,
        n_estimators=500, random_state=SEED, n_jobs=-1,
    )
    rf.fit(matrix(train), train["y"].to_numpy())
    return rf.predict(matrix(test))


MODELS = {
    "Random Forest (notebook features)": rf_baseline,
    "TF-IDF 1-2 gram + logistic regression": linear_model("lr"),
    "TF-IDF 1-2 gram + linear SVM": linear_model("svm"),
    "Lexicon features + logistic regression": linear_model("lr", ngrams=False, lex_cols=lexicon.FEATURES),
    "Length only + logistic regression": linear_model("lr", ngrams=False, lex_cols=lexicon.GROUPS["length"]),
}


def ablations() -> dict[str, Callable]:
    out = {
        "n-grams only": linear_model("lr"),
        "n-grams + all lexicon groups": linear_model("lr", lex_cols=lexicon.FEATURES),
        "n-grams with pronoun tokens removed": linear_model("lr", drop_pronoun_tokens=True),
        "lexicon, all groups": linear_model("lr", ngrams=False, lex_cols=lexicon.FEATURES),
    }
    for group, cols in lexicon.GROUPS.items():
        rest = [c for c in lexicon.FEATURES if c not in cols]
        out[f"lexicon without {group}"] = linear_model("lr", ngrams=False, lex_cols=rest)
    for group, cols in lexicon.GROUPS.items():
        out[f"lexicon, {group} only"] = linear_model("lr", ngrams=False, lex_cols=cols)
    return out
