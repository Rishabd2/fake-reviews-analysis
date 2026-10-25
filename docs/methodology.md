# Methodology

All numbers referenced here are in `results/tables.md` and `results/summary.json`, produced by
the commands in the README's Reproduce section. This file explains how they were produced.

## Question

Can a detector separate three kinds of hotel review: truthful guest reviews, fakes written by
paid crowd workers (Ott et al. 2011, 2013), and fakes written by an LLM (GPT-4o-mini)? And how
much of that separation comes from the text's style rather than from traces of how each set was
collected or generated?

## Data

See `docs/DATA.md` for sources, licenses, and how the LLM reviews were generated.

| Class | id | n |
| --- | --- | --- |
| truthful | 0 | 800 |
| crowdsourced fake (Mechanical Turk) | 1 | 800 |
| LLM fake (GPT-4o-mini) | 2 | 400 |

The binary task uses classes 0 and 1 only (n = 1,600). The three-way task uses all 2,000.

## Evaluation protocol

- **Folds.** Ott v1.4 ships five folds. Each fold contains every review for 4 of the 20 hotels, so
  a model never sees the test hotels during training. LLM reviews name no hotel; file number
  `i` goes to fold `(i - 1) mod 5 + 1`, which puts 40 positive and 40 negative LLM reviews in
  every fold.
- **Cross-validation.** Each fold is the test set once; the model is refit on the other four.
  Predictions from the five test folds are pooled (every review predicted exactly once) and
  metrics are computed on the pool.
- **Fold 5 held out.** For the binary task the tables also report the fold-5 test set alone
  (train on folds 1 to 4, n_test = 320), which is the same 80/20 size as the original notebook
  but hotel-disjoint.
- **Confidence intervals.** 95% percentile bootstrap, 2,000 resamples of the pooled predictions,
  seed 0. The intervals cover test-set sampling only, not training variance across seeds.
- **Metrics.** Accuracy; macro precision, recall, and F1 (unweighted mean over classes); for
  the binary task, precision, recall, and F1 of the deceptive class; per-class recall and
  confusion matrices for the three-way task. Macro F1 is the headline three-way metric because
  the classes are 800 / 800 / 400.
- **Hyperparameters.** Linear models pick C from {0.1, 1, 10, 100} by 3-fold stratified CV inside
  each training split (macro F1). The Random Forest keeps the saved notebook's grid-search result.
  DistilBERT uses fixed settings (below) with no tuning on test folds.
- **Seeds.** All randomness uses seed 0.

## Two views of the text

The LLM reviews carry traces of the prompt and of the generation pipeline that have nothing to
do with how a fake reads to a person: every one starts with one of six fixed clauses and
contains one of six required phrases, none names a hotel (78.8% of Ott reviews do), every one
is 88 to 106 words long (Ott: 25 to 784), and about half contain characters that never occur
in the ASCII-only Ott files (curly apostrophes, "é", long dashes). A detector can score well on
those traces alone.

- **Raw view:** reviews as written.
- **Controlled view** (`corpus.controlled_text`, applied identically to all three classes):
  fold to ASCII, drop the first sentence, replace the six seed phrases and all hotel names and
  "Chicago" with the same placeholder token "X" (so no class is left with grammatical holes the
  others lack), then keep the first N words, where N is drawn uniformly from 30 to 60 by an MD5
  hash of the file id. The cap does not depend on the class, so LLM reviews no longer share one
  length. Mean controlled length is 43.6 words for truthful, 43.7 for crowdsourced, and 44.9 for
  LLM reviews; the only remaining length signal is that some Ott reviews are shorter than their
  cap (minimum 1 word) while every LLM review reaches it. The "length only" model measures what
  that residual is worth. The placeholder itself is not perfectly balanced: every LLM review
  contained a required seed phrase, while 78.8% of Ott reviews name a hotel, so "x" appears among
  the top LLM coefficients (`results/top_ngrams.csv`).

The controlled view is a stricter test, not a cleaned dataset: it removes signal that a real
detector could legitimately use (for example, real reviews often name the hotel). Reporting both
shows how much of the raw score depends on these traces.

## Models

1. **Random Forest (notebook features).** The saved notebook's pipeline: lowercase, expand
   contractions, strip punctuation, WordNet-lemmatize, remove stop words; keep unigrams with count
   16 or more, and bigrams (count 16 or more) and trigrams (count 10 or more) with positive PMI,
   all counted on the training split; add the polarity rating and six counts (words, core words,
   stop words, characters, characters per word, first-person singular pronouns). Forest: 500
   trees, max depth 20, 5 features per split.
2. **TF-IDF + logistic regression / linear SVM.** Word unigrams and bigrams, min document
   frequency 2, sublinear TF, single-letter tokens kept (so "i" counts).
3. **Lexicon features + logistic regression.** 21 standardized features in seven groups
   (`src/lexicon.py`): pronouns (first singular, first plural, second, third), function words
   (articles, prepositions, auxiliaries, conjunctions, negations), affect (positive and negative
   emotion words, exclamation rate), spatial words, social context words (family, business,
   trip), length (words, characters, mean word length, sentences, mean sentence length), and
   diversity (type-token ratio of the first 50 tokens, comma rate). The categories follow LIWC
   groups that Ott et al. found informative, but the word lists are short open lists written for
   this repo. They are not LIWC.
4. **Length only.** The five length features. This is a floor that shows how much a model can do
   from length alone.
5. **DistilBERT fine-tune** (`distilbert-base-uncased`, `src/transformer.py`). Max 256 tokens,
   batch 16, AdamW at 5e-5 with weight decay 0.01, 3 epochs, linear schedule with 10% warmup,
   gradient clipping at 1.0, last-epoch model evaluated. Trained on Apple MPS. One seed per fold.

## Ablations

Logistic regression on the same folds:

- n-grams only, n-grams plus all lexicon groups, n-grams with pronoun tokens removed from the
  vocabulary;
- lexicon features with each group removed in turn, and each group alone.

## Cross-domain check

Train on all positive-polarity reviews, test on all negative ones, and the reverse. For the
three-way task, the LLM reviews are split by the polarity they were prompted with. This tests
whether cues learned on one sentiment transfer to the other; it does not test other cities,
other platforms, or other generators.

The classical models run the cross-domain check for both tasks and both views. DistilBERT was
not run on it (`src/transformer.py --setting cross_domain` supports it). For the binary task
DistilBERT was run on the fold-5 split only.

## Reproducing the saved notebook

`src/reproduce_notebook.py` repeats the saved bag-of-words notebook exactly: the notebook's row
order (`data/notebook_row_order.txt`), `train_test_split(test_size=0.2, random_state=0)`, and
the Random Forest settings. It then repeats the split with seeds 1 to 20. That split is random
over reviews, so the same hotel appears in train and test, unlike the fold protocol above.

## Limitations

- One generator (GPT-4o-mini), one prompt template, one temperature, one domain (Chicago hotels,
  reviews collected in 2011 to 2013). Results say nothing about other LLMs or prompts.
- The LLM reviews were filtered at generation time by a classifier trained on all Ott reviews
  (`src/review_eval.py`). That filter is part of how the data was made and could make the LLM
  class easier or harder to separate from crowdsourced fakes.
- The controlled view removes known artifacts but cannot rule out unknown ones (topic choice
  driven by the persona list, for example).
- LLM prompts named no hotel; Mechanical Turk workers were given one (Ott et al.), so hotel
  names are a collection difference, not a stylistic one.
- Bootstrap CIs ignore training variance; DistilBERT was run with one seed.
- The lexicon is a small open approximation of LIWC categories.
