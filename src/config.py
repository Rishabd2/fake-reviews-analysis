"""Paths and generation defaults. Azure credentials come from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_FILE = DATA_DIR / "cached_dataset.pkl"

# Ott et al. Deceptive Opinion Spam Corpus, downloaded separately.
# Point this at the extracted op_spam_v1.4 directory.
OP_SPAM_DIR = Path(
    os.getenv("OP_SPAM_DIR", str(DATA_DIR / "op_spam_v1.4"))
)

AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

PERSONAS = {
    "family": "a family with two kids on vacation",
    "business": "a frequent business traveler",
    "couple": "a couple celebrating an anniversary",
    "budget": "a budget-conscious backpacker",
}

# Filled by data_loader from truthful review lengths when the corpus is present.
POS_LEN_MEAN = 0
POS_LEN_SD = 0
NEG_LEN_MEAN = 0
NEG_LEN_SD = 0
