"""Tests that run without the Ott corpus, Azure, or a GPU."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import corpus  # noqa: E402
import lexicon  # noqa: E402
from evaluate import bootstrap, point_metrics  # noqa: E402
from linguistic_feature import count_pronoun  # noqa: E402


def test_first_person_count():
    words = "i loved my room and we enjoyed our stay".split()
    assert count_pronoun(words, ["i", "me", "my", "mine", "myself"]) == 2
    assert count_pronoun(words, ["we", "our", "us", "ours", "ourselves"]) == 2


def test_llm_reviews_complete_and_folded():
    df = corpus.load_llm()
    assert len(df) == 400
    assert (df.groupby("polarity").size() == 200).all()
    assert sorted(df["fold"].unique()) == [1, 2, 3, 4, 5]
    assert (df.groupby("fold").size() == 80).all()
    assert df["text"].str.split().str.len().between(80, 110).all()


def test_controlled_view_strips_prompt_artifacts():
    text = ("When I arrived at the hotel, the lobby was bright. The Hilton Chicago had a "
            "river view and the staff were kind—mostly. " + "word " * 100)
    out = corpus.controlled_text(text)
    assert not out.lower().startswith("when i arrived")
    for gone in ("hilton", "chicago", "river view"):
        assert gone not in out.lower()
    assert all(ord(c) < 128 for c in out)
    assert len(out.split()) == corpus.CONTROLLED_MAX_WORDS


def test_lexicon_features_shape_and_rates():
    f = lexicon.features_for("I loved my room. We will return!")
    assert set(f) == set(lexicon.FEATURES)
    assert f["first_sing"] == pytest.approx(2 / 7)
    assert f["first_plural"] == pytest.approx(1 / 7)
    assert f["n_sentences"] == 2


def test_point_metrics_known_values():
    y = [0, 0, 1, 1]
    p = [0, 1, 1, 1]
    m = point_metrics(y, p, positive=1)
    assert m["accuracy"] == 0.75
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 1.0
    assert m["macro_recall"] == pytest.approx(0.75)


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.8, y, rng.integers(0, 3, 300))
    out = bootstrap(y, p, n_boot=200)
    for v in out.values():
        assert v["ci_low"] <= v["value"] <= v["ci_high"]


def test_committed_summary_consistent_with_predictions():
    summary_path = ROOT / "results" / "summary.json"
    preds = sorted((ROOT / "results" / "predictions").glob("*.csv"))
    if not summary_path.exists() or not preds:
        pytest.skip("results not generated")
    summary = json.loads(summary_path.read_text())
    df = pd.concat([pd.read_csv(p) for p in preds])
    name = "TF-IDF 1-2 gram + logistic regression"
    g = df[(df.task == "three_way") & (df.view == "raw") & (df.setting == "cv") & (df.model == name)]
    acc = float((g.pred == g.y).mean())
    stored = summary["three_way"]["raw"]["cv"][name]["metrics"]["accuracy"]["value"]
    assert acc == pytest.approx(stored)
    assert len(g) == 2000
