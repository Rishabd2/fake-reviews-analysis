"""Fine-tune DistilBERT (distilbert-base-uncased) on the same folds as the classical models.

Settings: max 256 tokens, batch 16, AdamW lr 5e-5, weight decay 0.01, 3 epochs,
linear schedule with 10% warmup, seed 0, no early stopping and no validation-based
model selection (the last epoch is evaluated). Runs on Apple MPS or CPU.

    python src/transformer.py --task binary --view raw
    python src/transformer.py --task three_way --view controlled
    python src/transformer.py --task three_way --view raw --setting cross_domain

Appends to results/predictions/transformer.csv.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import REPO, apply_view, load_corpus  # noqa: E402

MODEL = "distilbert-base-uncased"
MAX_LEN, BATCH, LR, EPOCHS, SEED = 256, 16, 5e-5, 3, 0
OUT = REPO / "results" / "predictions" / "transformer.csv"
NAME = "DistilBERT fine-tune"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(tok, texts):
    """Pad to the longest review in the set (at most MAX_LEN), then trim each batch to its own longest."""
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="longest", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def trim(ids, mask):
    width = int(mask.sum(dim=1).max())
    return ids[:, :width], mask[:, :width]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, n_classes: int) -> np.ndarray:
    seed_all(SEED)
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=n_classes).to(dev)
    ids, mask = encode(tok, train["text"])
    labels = torch.tensor(train["y"].to_numpy(), dtype=torch.long)
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(ids, mask, labels), batch_size=BATCH, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for _ in range(EPOCHS):
        for b_ids, b_mask, b_y in loader:
            b_ids, b_mask = trim(b_ids, b_mask)
            out = model(input_ids=b_ids.to(dev), attention_mask=b_mask.to(dev), labels=b_y.to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    t_ids, t_mask = encode(tok, test["text"])
    preds = []
    with torch.no_grad():
        for i in range(0, len(t_ids), 64):
            b_ids, b_mask = trim(t_ids[i:i + 64], t_mask[i:i + 64])
            logits = model(input_ids=b_ids.to(dev), attention_mask=b_mask.to(dev)).logits
            preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def rows(df, pred, task, view, setting):
    return pd.DataFrame({
        "task": task, "view": view, "setting": setting, "model": NAME,
        "source": df["source"].to_numpy(), "fold": df["fold"].to_numpy(),
        "polarity": df["polarity"].to_numpy(), "y": df["y"].to_numpy(), "pred": pred,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "three_way"], required=True)
    parser.add_argument("--view", choices=["raw", "controlled"], default="raw")
    parser.add_argument("--setting", choices=["cv", "cross_domain"], default="cv")
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    df = apply_view(load_corpus(args.task), args.view)
    n_classes = df["y"].nunique()
    parts = []
    if args.setting == "cv":
        for fold in args.folds:
            t0 = time.time()
            test = df["fold"] == fold
            pred = fit_predict(df[~test], df[test], n_classes)
            acc = (pred == df.loc[test, "y"].to_numpy()).mean()
            print(f"{args.task} {args.view} fold {fold}: acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
            parts.append(rows(df[test], pred, args.task, args.view, "cv"))
    else:
        for src, dst in (("positive", "negative"), ("negative", "positive")):
            train, test = df[df["polarity"] == src], df[df["polarity"] == dst]
            pred = fit_predict(train, test, n_classes)
            acc = (pred == test["y"].to_numpy()).mean()
            print(f"{args.task} {args.view} {src}->{dst}: acc={acc:.3f}", flush=True)
            parts.append(rows(test, pred, args.task, args.view, f"train_{src}_test_{dst}"))

    new = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_csv(OUT)
        keys = set(zip(new["task"], new["view"], new["setting"], new["source"]))
        old = old[[k not in keys for k in zip(old["task"], old["view"], old["setting"], old["source"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
