"""Metrics with percentile bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

N_BOOT = 2000
SEED = 0


def _from_confusion(cm: np.ndarray, positive: int | None) -> dict:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    present = true > 0
    out = {
        "accuracy": tp.sum() / cm.sum(),
        "macro_precision": prec[present].mean(),
        "macro_recall": rec[present].mean(),
        "macro_f1": f1[present].mean(),
    }
    if positive is not None:
        out.update({"precision": prec[positive], "recall": rec[positive], "f1": f1[positive]})
    return out


def point_metrics(y_true, y_pred, positive: int | None = None, n_classes: int | None = None) -> dict:
    """Accuracy plus macro P/R/F1. For binary tasks also P/R/F1 of the `positive` class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    return _from_confusion(cm, positive)


def bootstrap(y_true, y_pred, positive: int | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Point estimate and 95% percentile CI for every metric in point_metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    point = point_metrics(y_true, y_pred, positive, n_classes)
    draws = {name: [] for name in point}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = point_metrics(y_true[idx], y_pred[idx], positive, n_classes)
        for name, v in m.items():
            draws[name].append(v)
    return {
        name: {"value": float(point[name]),
               "ci_low": float(np.percentile(draws[name], 2.5)),
               "ci_high": float(np.percentile(draws[name], 97.5))}
        for name in point
    }


def confusion(y_true, y_pred, n_classes: int) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()


def fmt(m: dict) -> str:
    return f"{100 * m['value']:.1f} [{100 * m['ci_low']:.1f}, {100 * m['ci_high']:.1f}]"


def cluster_bootstrap(y_true, y_pred, clusters, positive: int | None = None,
                      n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95% CI that resamples whole clusters (hotels) instead of single reviews.

    Ott reviews of the same hotel share vocabulary, so the i.i.d. bootstrap can be too
    narrow. Each Ott hotel is one cluster; each LLM review (no hotel) is its own cluster.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    clusters = np.asarray(clusters)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    point = point_metrics(y_true, y_pred, positive, n_classes)
    draws = {name: [] for name in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[k] for k in pick])
        m = point_metrics(y_true[idx], y_pred[idx], positive, n_classes)
        for name, v in m.items():
            draws[name].append(v)
    return {
        name: {"value": float(point[name]),
               "ci_low": float(np.percentile(draws[name], 2.5)),
               "ci_high": float(np.percentile(draws[name], 97.5))}
        for name in point
    }
