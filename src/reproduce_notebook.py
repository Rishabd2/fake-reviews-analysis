"""Re-run the saved bag-of-words notebook's Random Forest exactly as written.

Protocol from notebooks/bag_of_words.ipynb: one random 80/20 split
(train_test_split, random_state=0) over the 1,600 Ott reviews in the notebook's
row order (data/notebook_row_order.txt), n-gram vocabulary from the training
split, RandomForestClassifier(max_depth=20, max_features=5, n_estimators=500,
random_state=0). It then repeats the split with seeds 1 to 20 to show how much
a single random split moves the number.

    python src/reproduce_notebook.py

Writes results/notebook_reproduction.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_prep  # noqa: E402
import ngram  # noqa: E402
from corpus import REPO, load_ott  # noqa: E402
from evaluate import bootstrap  # noqa: E402
from features import add_features  # noqa: E402

ORDER = REPO / "data" / "notebook_row_order.txt"
OUT = REPO / "results" / "notebook_reproduction.json"


def notebook_frame() -> pd.DataFrame:
    ott = load_ott().set_index("source")
    order = ORDER.read_text().split()
    ott = ott.loc[order].reset_index()
    ott = add_features(ott)
    ott["Label"] = np.where(ott["label"] == "human_fake", 1, -1)
    ott["Rating"] = np.where(ott["polarity"] == "positive", 5, 1)
    return ott


def run_split(df: pd.DataFrame, seed: int) -> dict:
    tr, te = train_test_split(df, test_size=0.2, random_state=seed)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    uni, bi, tri = ngram.find_universal_ngrams(tr["PP_Review"], 16, 16, 10)
    sets = (set(uni), set(bi), set(tri))
    for part in (tr, te):
        part["Ngram"] = part["Word_List"].apply(lambda w: ngram.find_ngrams(w, *sets))
    vocab = uni + [" ".join(b) for b in bi] + [" ".join(t) for t in tri]
    word2ind = dict(zip(vocab, range(len(vocab))))
    X_tr, y_tr = data_prep.df2matrix(tr, word2ind)
    X_te, y_te = data_prep.df2matrix(te, word2ind)
    rf = RandomForestClassifier(
        random_state=0, criterion="gini", max_depth=20, max_features=5,
        max_samples=None, n_estimators=500, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    pos = (X_te["Rating"] == 5).to_numpy()
    y = y_te.to_numpy()
    return {
        "seed": seed,
        "train_accuracy": float(rf.score(X_tr, y_tr)),
        "test_accuracy": float((pred == y).mean()),
        "test_positive_accuracy": float((pred[pos] == y[pos]).mean()),
        "test_negative_accuracy": float((pred[~pos] == y[~pos]).mean()),
        "y": ((y == 1).astype(int)).tolist(),
        "pred": ((pred == 1).astype(int)).tolist(),
    }


def main() -> None:
    df = notebook_frame()
    first = run_split(df, 0)
    ci = bootstrap(first.pop("y"), first.pop("pred"), positive=1)
    print(f"seed 0: test acc {first['test_accuracy']:.3f} (notebook printed 0.903)")
    others = []
    for seed in range(1, 21):
        r = run_split(df, seed)
        r.pop("y"), r.pop("pred")
        others.append(r)
        print(f"seed {seed:2d}: test acc {r['test_accuracy']:.3f}")
    accs = np.array([r["test_accuracy"] for r in others])
    OUT.write_text(json.dumps({
        "protocol": "notebook 80/20 random split, n_test=320, notebook row order",
        "notebook_printed": {"test": 0.903, "test_positive": 0.904, "test_negative": 0.902, "train": 0.998},
        "seed0": first,
        "seed0_bootstrap": ci,
        "other_seeds": others,
        "other_seeds_summary": {
            "mean": float(accs.mean()), "std": float(accs.std(ddof=1)),
            "min": float(accs.min()), "max": float(accs.max()),
        },
    }, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
