# Notebooks

These are the original runs, copied from local work. Paths inside still assume the older `Chicago_Hotel_Review/` layout. After you download Ott v1.4, either recreate that folder or change the import to `data_prep.import_chi_review(os.environ["OP_SPAM_DIR"])`.

| File | What it did |
| --- | --- |
| `exploratory_data_analysis.ipynb` | Pronoun and length plots on the 1,600-review corpus |
| `bag_of_words.ipynb` | Random Forest 90.3% test, SVM+PCA 86.6% test |
| `bert_transfer.ipynb` | TensorFlow Hub BERT; saved class-wise rates 90.91% / 93.55% |
| `bert_transfer_pytorch.ipynb` | PyTorch BERT fine-tune (30 epochs logged, per-epoch accuracy not saved) |
| `glove_lstm_conv1d.ipynb` | Three Keras models on a 1200/400 split; best val 88.0% |

Do not re-run BERT in CI. The figures in `results/figures/` are redrawn from `results/metrics.json` and the linguistic CSVs.
