"""Per-review sentiment, bigrams, and punctuation counts on top of ling_feature()."""

from __future__ import annotations

import string
from collections import Counter

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams

from linguistic_feature import ling_feature

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

SIA = SentimentIntensityAnalyzer()


def _basic_features(text: str) -> dict:
    if nlp is not None:
        tokens = [tok.text for tok in nlp(text) if not tok.is_punct and not tok.is_space]
    else:
        tokens = text.split()
    return dict(
        sentiment=SIA.polarity_scores(text)["compound"],
        ngram_big=list(ngrams(tokens, 2)) if len(tokens) >= 2 else [],
        punctuation=Counter(ch for ch in text if ch in string.punctuation),
    )


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add linguistic columns, then VADER sentiment and punctuation stats."""
    df2 = ling_feature(df)
    stats = df2["Ori_Review"].apply(_basic_features)
    return df2.join(pd.DataFrame(stats.tolist(), index=df2.index))
