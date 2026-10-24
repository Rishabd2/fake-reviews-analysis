"""Load the Ott Chicago hotel corpus and clean it for bag-of-words models."""

from __future__ import annotations

import glob
import re
from itertools import zip_longest
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy.sparse import coo_matrix

from contractions import gen_contractions


def import_chi_review(data_root: str | Path | None = None) -> pd.DataFrame:
    """Read the four polarity/truthfulness folders of op_spam_v1.4.

    Returns columns:
      Label: 1 deceptive, -1 truthful
      Rating: 1 negative, 5 positive
      Ori_Review: original text
    """
    if data_root is None:
        from config import OP_SPAM_DIR

        data_root = OP_SPAM_DIR
    data_root = Path(data_root)

    folders = {
        "neg_dec": data_root / "negative_polarity" / "deceptive_from_MTurk",
        "neg_tru": data_root / "negative_polarity" / "truthful_from_Web",
        "pos_dec": data_root / "positive_polarity" / "deceptive_from_MTurk",
        "pos_tru": data_root / "positive_polarity" / "truthful_from_TripAdvisor",
    }

    lists = {
        key: sorted(glob.glob(str(path / "fold*" / "*.txt")))
        for key, path in folders.items()
    }

    texts = {key: [] for key in folders}
    for dec, tru, pdec, ptru in zip_longest(
        lists["neg_dec"], lists["neg_tru"], lists["pos_dec"], lists["pos_tru"]
    ):
        if None in (dec, tru, pdec, ptru):
            print("[Warning] file-count mismatch among polarity folders")
            continue
        for key, path in (
            ("neg_dec", dec),
            ("neg_tru", tru),
            ("pos_dec", pdec),
            ("pos_tru", ptru),
        ):
            with open(path, encoding="utf-8") as handle:
                texts[key].append(handle.read().strip())

    all_text = (
        texts["neg_dec"] + texts["neg_tru"] + texts["pos_dec"] + texts["pos_tru"]
    )
    label = (
        [1] * len(texts["neg_dec"])
        + [-1] * len(texts["neg_tru"])
        + [1] * len(texts["pos_dec"])
        + [-1] * len(texts["pos_tru"])
    )
    rating = [1] * (len(texts["neg_dec"]) + len(texts["neg_tru"])) + [5] * (
        len(texts["pos_dec"]) + len(texts["pos_tru"])
    )

    if not all_text:
        raise FileNotFoundError(
            f"No review files under {data_root}. See data/README.md."
        )

    return pd.DataFrame(
        {"Label": label, "Rating": rating, "Ori_Review": all_text}
    )


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize(words):
    lemmatized_words = []
    lm = WordNetLemmatizer()
    for word, pos in nltk.pos_tag(words):
        lemmatized_words.append(lm.lemmatize(word, get_wordnet_pos(pos)))
    return lemmatized_words


def remove_stopwords(words):
    stop_words = stopwords.words("english")
    stop_words.extend(
        ["chicago", "hotel", "would", "could", "should", "might", "room", "stay"]
    )
    return [word for word in words if word not in stop_words]


def preprocess_ngram(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, expand contractions, strip punctuation, lemmatize, drop stopwords."""
    df_pp = df.copy()
    clean_review = df_pp["Ori_Review"].str.lower()
    contraction_dict = gen_contractions()
    for con, full in contraction_dict.items():
        clean_review = clean_review.apply(lambda x, c=con, f=full: re.sub(c, f, x))
    clean_review = clean_review.apply(lambda x: re.sub(r"[,/.!?]", "", x))
    words = clean_review.apply(lambda x: simple_preprocess(x, deacc=True, min_len=1))
    words = words.apply(lemmatize)
    df_pp["Clean_Review"] = words.apply(lambda x: " ".join(x))
    words = words.apply(remove_stopwords)
    df_pp["PP_Review"] = words.apply(lambda x: " ".join(x))
    df_pp["Word_List"] = words
    return df_pp.reset_index(drop=True)


def df2matrix(df, word2ind):
    """Build a bag-of-words matrix plus the linguistic columns used by Random Forest."""
    rows = []
    cols = []
    for r, document in enumerate(df["Ngram"].tolist()):
        for word in document:
            rows.append(r)
            cols.append(word2ind[word])
    vals = np.array([1] * len(rows))
    X = coo_matrix((vals, [rows, cols]), shape=(max(rows) + 1, len(word2ind))).toarray()
    X = pd.DataFrame(X, columns=list(word2ind.keys()))
    X["Rating"] = df["Rating"].copy()
    ling_fea = [
        "num_word",
        "num_coreword",
        "num_stopword",
        "num_char",
        "char_per_word",
        "num_first_sing",
    ]
    X[ling_fea] = df[ling_fea].copy()
    y = df["Label"].copy()
    return X, y
