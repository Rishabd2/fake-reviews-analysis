"""Cache the Ott corpus as a list of {review, polarity, truthfulness} records."""

from __future__ import annotations

import pickle
from statistics import mean, stdev
from typing import Dict, List

import config
from data_prep import import_chi_review


def _build_dataset() -> List[Dict]:
    df = import_chi_review()
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "review": row.Ori_Review,
                "polarity": "positive" if row.Rating == 5 else "negative",
                "truthfulness": "deceptive" if row.Label == 1 else "truthful",
            }
        )

    pos_truthful_lengths = [
        len(r["review"].split())
        for r in records
        if r["polarity"] == "positive" and r["truthfulness"] == "truthful"
    ]
    neg_truthful_lengths = [
        len(r["review"].split())
        for r in records
        if r["polarity"] == "negative" and r["truthfulness"] == "truthful"
    ]
    if pos_truthful_lengths:
        config.POS_LEN_MEAN = int(mean(pos_truthful_lengths))
        config.POS_LEN_SD = int(stdev(pos_truthful_lengths)) if len(pos_truthful_lengths) > 1 else 0
    if neg_truthful_lengths:
        config.NEG_LEN_MEAN = int(mean(neg_truthful_lengths))
        config.NEG_LEN_SD = int(stdev(neg_truthful_lengths)) if len(neg_truthful_lengths) > 1 else 0
    return records


def load_dataset(force_reload: bool = False):
    if config.CACHE_FILE.exists() and not force_reload:
        with open(config.CACHE_FILE, "rb") as handle:
            return pickle.load(handle)

    print("[data_loader] Building dataset from", config.OP_SPAM_DIR)
    dataset = _build_dataset()
    config.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CACHE_FILE, "wb") as handle:
        pickle.dump(dataset, handle)
    print(f"[data_loader] Cached {len(dataset)} reviews to {config.CACHE_FILE}")
    return dataset
