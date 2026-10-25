# Results

Everything here is produced by committed scripts. No file in this folder contains review text.

| File | Produced by | What it holds |
| --- | --- | --- |
| `predictions/classical.csv.gz` | `src/run_experiments.py` | out-of-fold and cross-domain predictions for the classical models and ablations (file id, fold, true and predicted class) |
| `predictions/transformer.csv` | `src/transformer.py` | the same for DistilBERT |
| `predictions/prompt_word_check.csv` | `src/prompt_word_check.py` | TF-IDF LR on the controlled view with floor/view/lobby deleted |
| `summary.json` | `src/summarize.py` | metrics, 95% bootstrap CIs, confusion matrices, per-fold accuracy |
| `tables.md` | `src/summarize.py` | the same as Markdown tables |
| `notebook_reproduction.json` | `src/reproduce_notebook.py` | the saved notebook's Random Forest re-run on its own split, plus 20 other split seeds |
| `top_ngrams.csv` | `src/top_features.py` | largest logistic-regression coefficients per class (fit on all data, descriptive only) |
| `figures/three_way_macro_f1.png`, `binary_accuracy.png`, `confusion_matrices.png`, `ablations.png` | `src/make_figures.py` | README figures |

Older files from the original notebooks:

| File | Produced by | What it holds |
| --- | --- | --- |
| `metrics.json` | copied from saved notebook outputs and the upstream README | notebook-era scores (single random split) |
| `linguistic_summary.csv`, `linguistic_per_review.csv` | `src/compute_linguistic_stats.py` | Ott pronoun and length counts |
| `figures/model_accuracy.png`, `random_forest_split.png`, `neural_training_curves.png`, `linguistic_means.png`, `pronoun_histograms.png` | `src/plot_results.py` | notebook-era figures |
| `figures/original_pronoun_histogram.png` | upstream PauDK repo | EDA figure shipped upstream |
