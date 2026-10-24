"""Smoke tests that do not need the Ott corpus or Azure."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linguistic_feature import count_pronoun  # noqa: E402


def test_metrics_match_notebook_random_forest():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    rf = metrics["random_forest_detail"]
    assert rf["test"] == 0.903
    assert rf["test_positive"] == 0.904
    assert rf["test_negative"] == 0.902


def test_first_person_count():
    words = "i loved my room and we enjoyed our stay".split()
    assert count_pronoun(words, ["i", "me", "my", "mine", "myself"]) == 2
    assert count_pronoun(words, ["we", "our", "us", "ours", "ourselves"]) == 2


def test_samples_exist():
    samples = list((ROOT / "data" / "samples").glob("*.txt"))
    assert len(samples) == 4
    for path in samples:
        assert len(path.read_text().split()) > 20
