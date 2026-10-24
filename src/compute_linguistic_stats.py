"""Recompute Ott linguistic summaries used by the figures.

Writes means/SDs and per-review pronoun counts. Does not write review text.
Requires OP_SPAM_DIR or data/op_spam_v1.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_prep import import_chi_review, preprocess_ngram  # noqa: E402
from linguistic_feature import ling_feature  # noqa: E402

REPO = ROOT.parent
RESULTS = REPO / "results"
COLS = [
    "num_word",
    "num_coreword",
    "num_stopword",
    "num_char",
    "char_per_word",
    "num_first_sing",
    "num_first_pru",
    "num_third",
]


def main() -> None:
    df = import_chi_review()
    df = preprocess_ngram(df)
    df = ling_feature(df)
    df["label"] = df["Label"].map({1: "deceptive", -1: "truthful"})

    rows = []
    for label, group in df.groupby("label"):
        row = {"label": label, "n": len(group)}
        for col in COLS:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std())
            row[f"{col}_median"] = float(group[col].median())
        rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values("label")
    summary.to_csv(RESULTS / "linguistic_summary.csv", index=False)

    detail = df[["label", "num_first_sing", "num_first_pru", "num_word", "num_char"]].copy()
    detail.to_csv(RESULTS / "linguistic_per_review.csv", index=False)
    print(summary.to_string(index=False))
    print("Wrote", RESULTS / "linguistic_summary.csv")


if __name__ == "__main__":
    main()
