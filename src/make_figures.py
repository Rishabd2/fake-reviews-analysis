"""Draw the README figures from results/summary.json.

    python src/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import REPO  # noqa: E402
from summarize import MAIN_ORDER  # noqa: E402

RESULTS = REPO / "results"
FIG = RESULTS / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e4df"
SHORT = {
    "Random Forest (notebook features)": "Random Forest\n(notebook)",
    "TF-IDF 1-2 gram + logistic regression": "TF-IDF + LR",
    "TF-IDF 1-2 gram + linear SVM": "TF-IDF + SVM",
    "Lexicon features + logistic regression": "Lexicon + LR",
    "Length only + logistic regression": "Length only",
    "DistilBERT fine-tune": "DistilBERT",
}
CLASS_LABELS = ["truthful", "crowdsourced\nfake", "LLM fake"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def err(m):
    return [[m["value"] - m["ci_low"]], [m["ci_high"] - m["value"]]]


def bars(ax, groups, series, colors, labels, metric):
    models = [m for m in MAIN_ORDER if any(m in g for g in groups)]
    x = np.arange(len(models))
    w = 0.8 / len(groups)
    for i, (g, color, label) in enumerate(zip(groups, colors, labels)):
        for j, m in enumerate(models):
            if m not in g:
                continue
            v = g[m]["metrics"][metric]
            ax.bar(x[j] + (i - (len(groups) - 1) / 2) * w, 100 * v["value"], w * 0.92, color=color,
                   yerr=100 * np.array(err(v)), ecolor=INK, capsize=2, error_kw={"lw": 1},
                   label=label if j == 0 or (label and m == models[0]) else None)
            ax.text(x[j] + (i - (len(groups) - 1) / 2) * w, 100 * v["ci_high"] + 1, f"{100 * v['value']:.0f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [SHORT[m] for m in models])
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    return models


def hero(summary):
    t = summary["three_way"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars(ax, [t["raw"]["cv"], t["controlled"]["cv"]], None, [BLUE, ORANGE],
         ["reviews as written", "prompt artifacts removed"], "macro_f1")
    ax.axhline(100 / 3, color=MUTED, ls="--", lw=1)
    ax.text(5.45, 100 / 3 + 1, "chance", color=MUTED, fontsize=8, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Macro F1 (%), 5-fold CV, 95% CI")
    ax.set_title("Truthful vs crowdsourced fake vs GPT-4o-mini fake (n = 2,000)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "three_way_macro_f1.png", dpi=160)
    plt.close(fig)


def binary(summary, repro):
    b = summary["binary"]["raw"]["cv"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars(ax, [b], None, [BLUE], [None], "accuracy")
    ax.axhline(90.3, color=ORANGE, ls="--", lw=1)
    ax.text(5.45, 91, "saved notebook, single 80/20 split: 90.3", color=INK, fontsize=8, ha="right")
    ax.axhline(50, color=MUTED, ls=":", lw=1)
    ax.set_ylim(40, 105)
    ax.set_ylabel("Accuracy (%), 5-fold CV, 95% CI")
    ax.set_title("Truthful vs crowdsourced fake (Ott v1.4, n = 1,600)")
    fig.tight_layout()
    fig.savefig(FIG / "binary_accuracy.png", dpi=160)
    plt.close(fig)


def confusions(summary):
    picks = [m for m in ("TF-IDF 1-2 gram + logistic regression", "DistilBERT fine-tune")
             if m in summary["three_way"]["raw"]["cv"]]
    fig, axes = plt.subplots(len(picks), 2, figsize=(8.4, 3.7 * len(picks)), squeeze=False)
    for r, m in enumerate(picks):
        for c, (view, title) in enumerate((("raw", "as written"), ("controlled", "artifacts removed"))):
            ax = axes[r][c]
            cm = np.array(summary["three_way"][view]["cv"][m]["confusion"], float)
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{100 * norm[i, j]:.0f}%\n({int(cm[i, j])})", ha="center", va="center",
                            fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
            ax.set_xticks(range(3), CLASS_LABELS, fontsize=8)
            ax.set_yticks(range(3), CLASS_LABELS, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{SHORT[m]}, {title}", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Three-way confusion matrices (row-normalized, pooled over 5 folds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def ablation(summary):
    cols = [("binary", "raw", "binary, raw", BLUE), ("three_way", "raw", "three-way, raw", ORANGE),
            ("three_way", "controlled", "three-way, controlled", AQUA)]
    names = list(summary["binary"]["raw"]["ablation"].keys())
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(names) * 3 + 1))
    y = np.arange(len(names))
    h = 0.27
    for i, (task, view, label, color) in enumerate(cols):
        a = summary[task][view]["ablation"]
        vals = [100 * a[n]["metrics"]["macro_f1"]["value"] for n in names]
        lo = [100 * (a[n]["metrics"]["macro_f1"]["value"] - a[n]["metrics"]["macro_f1"]["ci_low"]) for n in names]
        hi = [100 * (a[n]["metrics"]["macro_f1"]["ci_high"] - a[n]["metrics"]["macro_f1"]["value"]) for n in names]
        ax.barh(y + (i - 1) * h, vals, h * 0.9, color=color, xerr=[lo, hi], ecolor=INK,
                error_kw={"lw": 0.8}, label=label)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Macro F1 (%), logistic regression, 5-fold CV, 95% CI")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Feature-group ablations")
    fig.tight_layout()
    fig.savefig(FIG / "ablations.png", dpi=160)
    plt.close(fig)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    repro_path = RESULTS / "notebook_reproduction.json"
    repro = json.loads(repro_path.read_text()) if repro_path.exists() else None
    hero(summary)
    binary(summary, repro)
    confusions(summary)
    ablation(summary)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
