"""Attach lexicon features and the Random Forest's preprocessed columns, with a disk cache.

Preprocessing is per review (no corpus statistics), so computing it once for all
rows does not leak across folds. The cache holds review text and is gitignored.
"""

from __future__ import annotations

import hashlib
import pickle

import pandas as pd

import lexicon
from corpus import DATA

CACHE = DATA / "cache"


def _key(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    from data_prep import preprocess_ngram
    from linguistic_feature import ling_feature

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"features_{_key(df['text'])}.pkl"
    if path.exists():
        with open(path, "rb") as handle:
            extra = pickle.load(handle)
    else:
        pp = preprocess_ngram(pd.DataFrame({"Ori_Review": df["text"].to_numpy()}))
        pp = ling_feature(pp)
        extra = pd.concat(
            [pp[["PP_Review", "Word_List", "num_word", "num_coreword", "num_stopword",
                 "num_char", "char_per_word", "num_first_sing"]],
             lexicon.feature_frame(df["text"])],
            axis=1,
        )
        with open(path, "wb") as handle:
            pickle.dump(extra, handle)
    out = df.reset_index(drop=True).copy()
    for col in extra.columns:
        out[col] = extra[col].to_numpy() if col != "Word_List" else list(extra[col])
    return out
