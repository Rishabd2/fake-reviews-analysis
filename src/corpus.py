"""Load the three-way corpus: truthful, crowdsourced fake, and LLM-generated fake.

Class ids:
  0 truthful     Ott v1.4 truthful reviews (TripAdvisor and other travel sites)
  1 human_fake   Ott v1.4 deceptive reviews written by Mechanical Turk workers
  2 llm_fake     GPT-4o-mini reviews in data/llm_generated (see docs/DATA.md)

Every row carries its Ott cross-validation fold (1 to 5). Ott folds are
hotel-disjoint: each fold holds 4 of the 20 hotels. LLM reviews name no hotel,
so they are assigned to folds round-robin by file number.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
LLM_DIR = DATA / "llm_generated"

CLASS_NAMES = ["truthful", "human_fake", "llm_fake"]

OTT_FOLDERS = {
    ("negative", "human_fake"): "negative_polarity/deceptive_from_MTurk",
    ("negative", "truthful"): "negative_polarity/truthful_from_Web",
    ("positive", "human_fake"): "positive_polarity/deceptive_from_MTurk",
    ("positive", "truthful"): "positive_polarity/truthful_from_TripAdvisor",
}

# The six openings and six seed phrases that src/generator.py forced into every
# LLM review. They are prompt artifacts, not style, so the controlled view drops them.
SEED_PHRASES = [
    "spacious room",
    "river view",
    "helpful staff",
    "dated décor",
    "dated decor",
    "musty smell",
    "weak coffee",
]

# Hotel names and the city. Ott reviews name the hotel; the LLM prompt never did.
NAME_TOKENS = [
    "affinia", "milenorth", "allegro", "kimpton", "amalfi", "ambassador",
    "conrad", "fairmont", "millennium", "hard rock", "hardrock",
    "hilton", "homewood", "hyatt", "regency", "intercontinental", "james",
    "knickerbocker", "monaco", "omni", "palmer", "sheraton", "sofitel",
    "swissotel", "talbott", "chicago",
]
_NAME_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in NAME_TOKENS) + r")\b", re.I)
_SEED_RE = re.compile("|".join(re.escape(p) for p in SEED_PHRASES), re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
# Ott v1.4 is plain ASCII; GPT-4o-mini output has curly quotes, accents, and long dashes.
_ASCII = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
          "\u2014": ", ", "\u2013": "-", "\u00e9": "e"}

CONTROLLED_MAX_WORDS = 60


def ott_dir() -> Path:
    return Path(os.getenv("OP_SPAM_DIR", str(DATA / "op_spam_v1.4")))


def load_ott(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root else ott_dir()
    rows = []
    for (polarity, label), sub in OTT_FOLDERS.items():
        for path in sorted((root / sub).glob("fold*/*.txt")):
            fold = int(path.parent.name.replace("fold", ""))
            hotel = path.stem.split("_")[1]
            rows.append(
                dict(
                    text=path.read_text(encoding="utf-8").strip(),
                    label=label,
                    polarity=polarity,
                    fold=fold,
                    hotel=hotel,
                    source=f"{sub}/{path.parent.name}/{path.name}",
                )
            )
    if len(rows) != 1600:
        raise FileNotFoundError(
            f"Expected 1600 Ott reviews under {root}, found {len(rows)}. "
            "Run: python scripts/fetch_data.py"
        )
    return pd.DataFrame(rows)


def load_llm(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root else LLM_DIR
    rows = []
    for polarity in ("positive", "negative"):
        for path in sorted((root / polarity).glob("*.txt")):
            idx = int(path.stem.split("_")[1])
            rows.append(
                dict(
                    text=path.read_text(encoding="utf-8").strip(),
                    label="llm_fake",
                    polarity=polarity,
                    fold=(idx - 1) % 5 + 1,
                    hotel="none",
                    source=f"llm_generated/{polarity}/{path.name}",
                )
            )
    return pd.DataFrame(rows)


def load_corpus(task: str = "three_way") -> pd.DataFrame:
    """task is 'binary' (Ott only) or 'three_way' (Ott plus LLM reviews)."""
    df = load_ott()
    if task == "three_way":
        df = pd.concat([df, load_llm()], ignore_index=True)
    elif task != "binary":
        raise ValueError(task)
    names = CLASS_NAMES if task == "three_way" else CLASS_NAMES[:2]
    df["y"] = df["label"].map({n: i for i, n in enumerate(names)})
    return df.reset_index(drop=True)


def controlled_text(text: str, max_words: int = CONTROLLED_MAX_WORDS) -> str:
    """Remove the artifacts the generation prompt introduced, for every class alike.

    0. Fold characters to ASCII (Ott v1.4 has none outside ASCII; the LLM text does).
    1. Drop the first sentence (the LLM reviews all open with one of six fixed clauses).
    2. Delete the six seed phrases the prompt required.
    3. Delete hotel names and "Chicago" (Ott reviews name hotels, LLM reviews do not).
    4. Keep the first `max_words` words (LLM output was capped at 110 words).
    """
    for src, dst in _ASCII.items():
        text = text.replace(src, dst)
    text = text.encode("ascii", "ignore").decode()
    sentences = _SENT_RE.split(text.strip())
    body = " ".join(sentences[1:]) if len(sentences) > 1 else text
    body = _SEED_RE.sub(" ", body)
    body = _NAME_RE.sub(" ", body)
    words = body.split()
    return " ".join(words[:max_words])


def apply_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """view is 'raw' or 'controlled'."""
    out = df.copy()
    if view == "controlled":
        out["text"] = out["text"].map(controlled_text)
    elif view != "raw":
        raise ValueError(view)
    return out
