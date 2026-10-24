"""Redraw published local results. Does not train models or invent scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
METRICS_PATH = RESULTS / "metrics.json"

TRUTHFUL = "#1f4e79"
DECEPTIVE = "#b42318"
MUTED = "#5c6b73"
ACCENT = "#2a6f4e"
GRID = "#e6e4df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


SHORT_NAMES = {
    "Human (Ott et al.)": "Human (cited)",
    "Ott et al. 2011/2013 baseline": "Ott et al. 2011/2013",
    "SVM + PCA (601 components, C=3)": "SVM + PCA",
    "GloVe + Conv1D + LSTM (best val epoch 30)": "GloVe + Conv1D + LSTM",
    "Random Forest (n-grams + lexicon)": "Random Forest",
    "BERT (PauDK reported test)": "BERT (reported)",
}


def plot_model_accuracy(metrics: dict, out: Path) -> None:
    rows = metrics["held_out_accuracy"]
    names = [SHORT_NAMES.get(r["model"], r["model"]) for r in rows]
    scores = [r["accuracy"] * 100 for r in rows]
    colors = [ACCENT if r.get("highlight") else MUTED for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Held-out accuracy (%)")
    ax.set_title("Detectors on Ott et al. Chicago hotel reviews")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            score + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_rf_split(metrics: dict, out: Path) -> None:
    rf = metrics["random_forest_detail"]
    labels = ["Train", "Test overall", "Test positive", "Test negative"]
    values = [
        rf["train"] * 100,
        rf["test"] * 100,
        rf["test_positive"] * 100,
        rf["test_negative"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [MUTED, ACCENT, TRUTHFUL, DECEPTIVE]
    ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Random Forest (n-grams + lexicon features)")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_training_curves(metrics: dict, out: Path) -> None:
    histories = metrics["neural_val_accuracy"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    palette = {
        "GloVe + Conv1D + LSTM": ACCENT,
        "Stacked LSTM": TRUTHFUL,
        "LSTM then Conv1D": DECEPTIVE,
    }
    for name, series in histories.items():
        epochs = np.arange(1, len(series) + 1)
        ax.plot(
            epochs,
            [v * 100 for v in series],
            label=name,
            color=palette.get(name, MUTED),
            lw=1.8,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Neural models from the GloVe/LSTM notebook (1200 / 400 split)")
    ax.set_ylim(50, 95)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_linguistic_means(summary_path: Path, out: Path) -> None:
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    features = [
        ("num_first_sing", "First-person singular"),
        ("num_first_pru", "First-person plural"),
        ("num_coreword", "Core words"),
        ("num_stopword", "Stop words"),
        ("num_word", "Words"),
        ("num_char", "Characters"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.2))
    for ax, (col, title) in zip(axes.ravel(), features):
        means = df.set_index("label")[f"{col}_mean"]
        labels = ["Truthful", "Deceptive"]
        vals = [means["truthful"], means["deceptive"]]
        ax.bar(labels, vals, color=[TRUTHFUL, DECEPTIVE], width=0.58)
        ymax = max(vals) * 1.18 if max(vals) else 1
        ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Mean per review")
    fig.suptitle(
        "Linguistic feature means on Ott v1.4 (n=800 per class)",
        fontsize=12,
        fontweight="semibold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pronoun_hists(detail_path: Path, out: Path) -> None:
    if not detail_path.exists():
        return
    df = pd.read_csv(detail_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    specs = [
        ("num_first_sing", "First-person singular (I, me, my)"),
        ("num_first_pru", "First-person plural (we, our, us)"),
    ]
    for ax, (col, title) in zip(axes, specs):
        for label, color, name in (
            ("truthful", TRUTHFUL, "Truthful"),
            ("deceptive", DECEPTIVE, "Deceptive"),
        ):
            vals = df.loc[df["label"] == label, col]
            ax.hist(
                vals,
                bins=range(0, min(int(vals.max()) + 2, 40)),
                histtype="stepfilled",
                alpha=0.35,
                color=color,
                label=name,
                edgecolor=color,
                lw=1.2,
            )
            ax.axvline(vals.mean(), color=color, ls="--", lw=1.1)
        ax.set_title(title)
        ax.set_xlabel("Count in review")
        ax.set_ylabel("Reviews")
        ax.legend(frameon=False)
    fig.suptitle("First-person singular is higher in deceptive reviews", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    plot_model_accuracy(metrics, FIGURES / "model_accuracy.png")
    plot_rf_split(metrics, FIGURES / "random_forest_split.png")
    plot_training_curves(metrics, FIGURES / "neural_training_curves.png")
    plot_linguistic_means(RESULTS / "linguistic_summary.csv", FIGURES / "linguistic_means.png")
    plot_pronoun_hists(RESULTS / "linguistic_per_review.csv", FIGURES / "pronoun_histograms.png")
    print("Wrote figures to", FIGURES)


if __name__ == "__main__":
    main()
