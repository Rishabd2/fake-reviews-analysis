# Fake reviews analysis

This repo detects crowdsourced fake hotel reviews on Ott et al.'s 1,600-review Chicago corpus and generates a second fake class with GPT-4o-mini. The best notebook result on the Ott task is a bag-of-words Random Forest at **90.3%** test accuracy. The PauDK writeup reports **92%** for fine-tuned BERT. Neither number is a score on the AI-generated reviews.

## Question

Given a hotel review, is it a real guest or a paid faker? After 2023 the faker can also be an LLM. The work has two parts:

1. Reproduce and slightly beat the Ott 87% n-gram baseline on Mechanical Turk deception.
2. Write persona-conditioned GPT-4o-mini reviews that can sit next to that corpus as adversarial examples.

A three-way detector (truthful / Turker / LLM) was planned. It is not in the saved notebooks.

## Data

| Split | n | Origin |
| --- | --- | --- |
| Truthful positive | 400 | TripAdvisor |
| Truthful negative | 400 | Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor, Yelp |
| Deceptive positive | 400 | Mechanical Turk |
| Deceptive negative | 400 | Mechanical Turk |

Download Ott v1.4 yourself. It is CC BY-NC-SA 3.0, so it is not in git. Instructions: [`data/README.md`](data/README.md).

Four GPT-4o-mini samples (2 positive, 2 negative) are in [`data/samples/`](data/samples/). The original run wrote about 400 such files.

## Features and models

Cleaning: lowercase, expand contractions, drop punctuation, WordNet lemmatize, remove stopwords. Features: unigram/bigram/trigram counts plus word count, core-word count, stop-word count, characters, and first-person singular pronouns.

Deceptive reviews use more first-person singular pronouns (mean 7.48 vs 4.68). A rerun of the same feature code finds truthful reviews slightly longer, which is the opposite of what the project notes said. Stop-word rates are close. See [`results/figures/pronoun_histograms.png`](results/figures/pronoun_histograms.png).

| Model | Held-out accuracy | Where the number came from |
| --- | --- | --- |
| Human (cited) | ~50% | PauDK README |
| Ott et al. 2011/2013 | 87% | papers / PauDK baseline |
| SVM + PCA (601 dim, C=3) | 86.6% | `notebooks/bag_of_words.ipynb` |
| GloVe + Conv1D + LSTM | 88.0% val | `notebooks/glove_lstm_conv1d.ipynb` epoch 30 |
| Random Forest | **90.3%** test (90.4% pos / 90.2% neg) | same bag-of-words notebook |
| BERT | 92% reported; notebook saved 90.91% truthful-as-truthful and 93.55% deceptive-as-deceptive | PauDK README + `notebooks/bert_transfer.ipynb` |

The TF-IDF logistic regression in `src/review_eval.py` is a generation filter fit on the full Ott set. Do not quote it as a test score.

## Generation

`src/generator.py` conditions GPT-4o-mini on polarity and one of four personas (family, business traveler, couple, budget backpacker). The first script, `src/archive/llm2.py`, used a single generic prompt and produced flatter text. Details: [`docs/methods.md`](docs/methods.md).

## How to run

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet averaged_perceptron_tagger vader_lexicon
# optional, only for spaCy paths
python -m spacy download en_core_web_sm
```

Redraw the committed figures (no corpus required):

```bash
python src/plot_results.py
```

Recompute linguistic means after you have Ott v1.4:

```bash
export OP_SPAM_DIR=/path/to/op_spam_v1.4
python src/compute_linguistic_stats.py
python src/plot_results.py
```

Generate new AI reviews (needs Azure OpenAI):

```bash
cp .env.example .env   # fill AZURE_* keys
export OP_SPAM_DIR=/path/to/op_spam_v1.4
python src/main.py
```

Bag-of-words and BERT training live in `notebooks/`. Those notebooks still look for a `Chicago_Hotel_Review/` folder. Point `import_chi_review` at `OP_SPAM_DIR` or recreate that layout.

## Limitations

- Reported accuracies are truthful vs Turker, on one city and one domain.
- The BERT 92% figure is from the upstream README. The saved TF notebook did not print an overall test accuracy.
- `review_eval.py` trains on all 1,600 Ott reviews. That is leakage if you treat it as a detector.
- The prompt-feedback loop and the human authenticity survey in the project notes were not finished.
- Hotel n-grams will not transfer to products or restaurants without new data.

## Layout

```
src/           loaders, features, RF/BERT helpers, generator
notebooks/     original executed analysis
docs/          methods and third-party credits
data/samples/  four AI reviews
results/       metrics.json and figures
```

## Credits

Classifier notebooks and the 90.3% / 92% writeup start from [PauDK/Deceptive-Review-Detection](https://github.com/PauDK/Deceptive-Review-Detection). The generation pipeline and this packaging are later local work. Full citations: [`docs/third_party.md`](docs/third_party.md).

MIT for original files. Ott data stays under CC BY-NC-SA 3.0.
