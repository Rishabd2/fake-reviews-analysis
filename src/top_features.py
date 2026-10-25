"""List the n-grams that most separate the three classes.

Fits the TF-IDF logistic regression (C=10) on all 2,000 reviews, once per view, and
writes the 15 largest coefficients per class to results/top_ngrams.csv. This is a
description of the fitted model, not a held-out score.

    python src/top_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import CLASS_NAMES, REPO, apply_view, load_corpus  # noqa: E402
from models import SEED, _tfidf  # noqa: E402

K = 15


def main() -> None:
    rows = []
    base = load_corpus("three_way")
    for view in ("raw", "controlled"):
        df = apply_view(base, view)
        vec = _tfidf()
        X = vec.fit_transform(df["text"])
        clf = LogisticRegression(C=10, max_iter=5000, random_state=SEED).fit(X, df["y"])
        vocab = np.array(vec.get_feature_names_out())
        for c, name in enumerate(CLASS_NAMES):
            top = np.argsort(clf.coef_[c])[::-1][:K]
            for rank, j in enumerate(top, 1):
                rows.append({"view": view, "class": name, "rank": rank, "ngram": vocab[j],
                             "coef": round(float(clf.coef_[c][j]), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results" / "top_ngrams.csv", index=False)
    for (view, name), g in out.groupby(["view", "class"], sort=False):
        print(f"{view:10s} {name:10s} " + ", ".join(g["ngram"]))


if __name__ == "__main__":
    main()
