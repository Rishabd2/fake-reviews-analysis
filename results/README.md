# Results

`metrics.json` holds the notebook and README numbers used by `src/plot_results.py`. Nothing in that file was estimated.

`linguistic_summary.csv` and `linguistic_per_review.csv` are recomputed from a local Ott v1.4 copy. They store counts only.

Figures:

- `model_accuracy.png` — held-out scores from `metrics.json`
- `random_forest_split.png` — train / test / polarity split for the 90.3% forest
- `neural_training_curves.png` — validation accuracy by epoch from the GloVe notebook
- `linguistic_means.png` — class means from the summary CSV
- `pronoun_histograms.png` — first-person counts per review
- `original_pronoun_histogram.png` — the EDA figure shipped with the PauDK repo
