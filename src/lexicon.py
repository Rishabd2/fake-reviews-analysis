"""LIWC-style lexicon features in open word lists.

LIWC itself is proprietary. These groups follow the categories Ott et al. found
useful (pronouns, spatial detail, affect, function words) with short hand-written
lists, so the numbers are not LIWC scores. Rates are per word.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

WORDS = {
    "first_sing": {"i", "me", "my", "mine", "myself"},
    "first_plural": {"we", "us", "our", "ours", "ourselves"},
    "second": {"you", "your", "yours", "yourself", "yourselves"},
    "third": {"he", "she", "they", "him", "her", "them", "his", "hers", "their",
              "theirs", "himself", "herself", "themselves"},
    "article": {"a", "an", "the"},
    "preposition": {"in", "on", "at", "to", "of", "for", "with", "by", "from",
                    "into", "onto", "over", "under", "near", "across", "through",
                    "about", "after", "before", "during", "around", "behind"},
    "auxiliary": {"is", "am", "are", "was", "were", "be", "been", "being", "have",
                  "has", "had", "do", "does", "did", "will", "would", "can",
                  "could", "should", "may", "might", "must"},
    "conjunction": {"and", "but", "or", "so", "because", "although", "though",
                    "while", "if", "yet"},
    "negation": {"no", "not", "never", "nothing", "nobody", "none", "nor",
                 "cannot", "dont", "didnt", "wasnt", "isnt", "wont", "couldnt"},
    "pos_emotion": {"great", "good", "love", "loved", "lovely", "amazing",
                    "excellent", "wonderful", "perfect", "beautiful", "best",
                    "nice", "happy", "enjoyed", "comfortable", "friendly",
                    "delightful", "fantastic", "awesome", "pleasant"},
    "neg_emotion": {"bad", "terrible", "horrible", "awful", "worst", "dirty",
                    "rude", "disappointed", "disappointing", "poor", "unhappy",
                    "angry", "annoyed", "upset", "disgusting", "uncomfortable",
                    "noisy", "broken", "smelly", "unacceptable"},
    "spatial": {"floor", "view", "lobby", "bathroom", "bed", "window", "street",
                "block", "location", "elevator", "corner", "upstairs",
                "downstairs", "small", "large", "big", "tiny", "walk", "distance",
                "close", "next", "above", "below", "inside", "outside"},
    "family_business": {"husband", "wife", "kids", "children", "family",
                        "business", "conference", "meeting", "work",
                        "anniversary", "vacation", "trip", "weekend"},
}

GROUPS = {
    "pronouns": ["first_sing", "first_plural", "second", "third"],
    "function_words": ["article", "preposition", "auxiliary", "conjunction", "negation"],
    "affect": ["pos_emotion", "neg_emotion", "exclaim_rate"],
    "spatial": ["spatial"],
    "social_context": ["family_business"],
    "length": ["n_words", "n_chars", "mean_word_len", "n_sentences", "mean_sent_len"],
    "diversity": ["ttr_50", "comma_rate"],
}
FEATURES = [f for group in GROUPS.values() for f in group]

_TOKEN_RE = re.compile(r"[a-z]+")
_SENT_RE = re.compile(r"[.!?]+")


def features_for(text: str) -> dict:
    lowered = text.lower().replace("'", "").replace("’", "")
    tokens = _TOKEN_RE.findall(lowered)
    n = max(len(tokens), 1)
    out = {name: sum(t in vocab for t in tokens) / n for name, vocab in WORDS.items()}
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    out["exclaim_rate"] = text.count("!") / n
    out["comma_rate"] = text.count(",") / n
    out["n_words"] = len(tokens)
    out["n_chars"] = len(text)
    out["mean_word_len"] = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    out["n_sentences"] = len(sentences)
    out["mean_sent_len"] = len(tokens) / max(len(sentences), 1)
    head = tokens[:50]
    out["ttr_50"] = len(set(head)) / max(len(head), 1)
    return {k: out[k] for k in FEATURES}


def feature_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([features_for(t) for t in texts], columns=FEATURES)
