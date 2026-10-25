"""Check whether the LLM class stays separable once the prompt's example words are gone.

The generation prompt asked for "at least one spatial detail (floor, view, lobby, etc.)".
Those words top the LLM class's coefficients even in the controlled view. This script
deletes floor/floors/view/views/lobby from every review (all classes) on top of the
controlled view and re-runs the TF-IDF logistic regression with the same folds.

    python src/prompt_word_check.py

Writes results/predictions/prompt_word_check.csv (picked up by src/summarize.py).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import REPO, apply_view, load_corpus  # noqa: E402
from models import MODELS  # noqa: E402
from run_experiments import cv_predictions, rows  # noqa: E402

WORDS = re.compile(r"\b(floors?|views?|lobby|lobbies)\b", re.I)
NAME = "TF-IDF 1-2 gram + logistic regression"


def main() -> None:
    df = apply_view(load_corpus("three_way"), "controlled")
    df["text"] = df["text"].map(lambda t: WORDS.sub(" ", t))
    pred = cv_predictions(df, MODELS[NAME])
    out = rows(df, pred, "three_way", "controlled_no_prompt_words", "cv", NAME)
    path = REPO / "results" / "predictions" / "prompt_word_check.csv"
    out.to_csv(path, index=False)
    llm = df["y"] == 2
    print(f"accuracy {(pred == df['y']).mean():.3f}, LLM recall {(pred[llm] == 2).mean():.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
