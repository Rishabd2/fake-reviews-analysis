"""Run the classical detectors and write out-of-fold predictions.

Protocol: Ott's five hotel-disjoint folds. Each fold is the test set once and the
other four are training data. LLM reviews are assigned to folds round-robin.
Cross-domain: train on one polarity (all folds), test on the other.

    python src/run_experiments.py            # everything
    python src/run_experiments.py --quick    # main models only, raw view

Writes results/predictions/classical.csv (no review text, only file ids).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import REPO, apply_view, load_corpus  # noqa: E402
from features import add_features  # noqa: E402
from models import MODELS, ablations  # noqa: E402

PRED_DIR = REPO / "results" / "predictions"
CROSS_MODELS = [
    "Random Forest (notebook features)",
    "TF-IDF 1-2 gram + logistic regression",
    "TF-IDF 1-2 gram + linear SVM",
    "Lexicon features + logistic regression",
]


def cv_predictions(df: pd.DataFrame, model) -> pd.Series:
    pred = pd.Series(index=df.index, dtype=int)
    for fold in sorted(df["fold"].unique()):
        test = df["fold"] == fold
        pred[test] = model(df[~test], df[test])
    return pred


def rows(df, pred, task, view, setting, name):
    return pd.DataFrame({
        "task": task, "view": view, "setting": setting, "model": name,
        "source": df["source"].to_numpy(), "fold": df["fold"].to_numpy(),
        "polarity": df["polarity"].to_numpy(), "y": df["y"].to_numpy(),
        "pred": pd.Series(pred).astype(int).to_numpy(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out = []
    views = ["raw"] if args.quick else ["raw", "controlled"]
    for task in ("binary", "three_way"):
        base = load_corpus(task)
        for view in views:
            df = add_features(apply_view(base, view))
            for name, model in MODELS.items():
                t0 = time.time()
                pred = cv_predictions(df, model)
                out.append(rows(df, pred, task, view, "cv", name))
                acc = (pred.to_numpy() == df["y"].to_numpy()).mean()
                print(f"{task:9s} {view:10s} cv  {name:42s} acc={acc:.3f} ({time.time() - t0:.0f}s)")
            if args.quick:
                continue
            for name, model in ablations().items():
                pred = cv_predictions(df, model)
                out.append(rows(df, pred, task, view, "ablation", name))
                acc = (pred.to_numpy() == df["y"].to_numpy()).mean()
                print(f"{task:9s} {view:10s} abl {name:42s} acc={acc:.3f}")
            for src, dst in (("positive", "negative"), ("negative", "positive")):
                train, test = df[df["polarity"] == src], df[df["polarity"] == dst]
                for name in CROSS_MODELS:
                    pred = MODELS[name](train, test)
                    out.append(rows(test, pred, task, view, f"train_{src}_test_{dst}", name))
                    acc = (pred == test["y"].to_numpy()).mean()
                    print(f"{task:9s} {view:10s} {src[:3]}->{dst[:3]} {name:42s} acc={acc:.3f}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / ("classical_quick.csv" if args.quick else "classical.csv")
    pd.concat(out, ignore_index=True).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
