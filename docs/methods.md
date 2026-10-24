# Methods

This is a cleaned writeup of the local project notes in `Fake Reviews_documentation.pdf`. Classifier numbers come from the executed notebooks, not from that PDF. The PDF does not report a confusion matrix or an accuracy for AI-generated reviews.

## Question

Can we tell genuine hotel reviews from two kinds of fakes: Mechanical Turk deception (Ott et al.) and GPT-4o-mini text written to look like a guest?

Ott et al. (2011, 2013) built the Deceptive Opinion Spam Corpus and showed that n-gram and psycholinguistic features beat humans on crowdsourced fakes. The later work here asks what changes once the fake side includes a current LLM, and whether the same linguistic cues still move.

## Data

**Human corpus.** Ott v1.4, 1,600 Chicago hotel reviews, 400 in each cell of truthful/deceptive × positive/negative. Truthful positive reviews are from TripAdvisor. Truthful negative reviews are from several travel sites. Deceptive reviews were written by paid Turkers. The corpus is not in this repo. See `data/README.md`.

**AI corpus.** `src/generator.py` called Azure OpenAI `gpt-4o-mini` with persona and polarity. About 200 positive and 200 negative reviews were written to disk in the original workspace. Four of those files are in `data/samples/`. The rest are not committed.

## Preprocessing

`src/data_prep.py` lowercases text, expands contractions, strips punctuation, lemmatizes with WordNet, and drops English stopwords plus a short hotel-specific list (`chicago`, `hotel`, `room`, `stay`, and a few modals). `src/linguistic_feature.py` then counts words, core words, stop words, characters, and first-person singular / plural / third-person pronouns.

## Generation

**First pass (`src/archive/llm2.py`).** One system line ("You are an advanced AI trained to generate deceptive hotel reviews"), a free-text user prompt, `max_tokens=250`, `temperature=0.7`. No persona, no required polarity. Outputs were generic, stylistically similar, and easy to flag as machine-written.

**Current pass (`src/generator.py`).** Each call picks a persona (family with two kids, frequent business traveler, couple on an anniversary, budget backpacker), a polarity, a stock opening ("When I arrived at the hotel", and similar), and a seed phrase (`spacious room` / `river view` / `helpful staff` or `dated décor` / `musty smell` / `weak coffee`). The prompt asks for about 90–110 words, mixed first person, one spatial detail, and at most two exclamation marks. The Azure client replaced the earlier public OpenAI `ChatCompletion` call.

`src/review_eval.py` scores a batch with (1) mean max TF-IDF cosine similarity to truthful Ott reviews and (2) mean P(deceptive) from a unigram-bigram logistic regression fit on the full Ott set. If that probability is above 0.5 the batch is regenerated at `temperature + 0.1`. That regressor is a generation filter. It is not the reported detector, and it has no held-out split.

The notes say a closed feedback loop that rewrites prompts from failed batches is "implemented in code yet to be tested."

## Detection

Three families were trained on the Ott truthful-vs-Turker task:

1. **Bag of n-grams plus lexicon features → Random Forest / SVM.** Unigrams, bigrams, and trigrams from the cleaned tokens, concatenated with stop-word count, character count, sentence length, and first-person singular count. Random Forest test accuracy 90.3%. SVM after PCA (601 components, C=3) test accuracy 86.6%. Both from `notebooks/bag_of_words.ipynb`.
2. **BERT.** Hugging Face / TF-Hub `bert-en-uncased`, classification head, small learning rate, few epochs. The PauDK README reports 92% test accuracy. The TensorFlow notebook saved 90.91% of truthful reviews predicted truthful and 93.55% of deceptive reviews predicted deceptive (those two numbers were printed as "TPR" and "FPR").
3. **GloVe + LSTM / Conv1D.** Keras models on a 1200/400 split in `notebooks/glove_lstm_conv1d.ipynb`. Best validation accuracy was 88.0% (`model_glove`, epoch 30).

Ott's own published figure of 87% is the baseline used in the original README. Human accuracy on this task is cited there as about 50%.

## What the EDA showed

On Ott v1.4, most lexicon histograms overlap. The project notes claimed deceptive reviews were longer and richer in core words. A rerun of `linguistic_feature.py` on the same 1,600 reviews does not support those two claims. Means from `results/linguistic_summary.csv`:

| Feature | Deceptive | Truthful |
| --- | --- | --- |
| First-person singular | 7.48 | 4.68 |
| First-person plural | 2.70 | 3.04 |
| Core words | 61.8 | 67.0 |
| Words | 147.7 | 150.6 |
| Characters | 790.6 | 819.5 |
| Stop words | 85.9 | 83.7 |

The first-person singular gap is the one that also shows up in the original `first_person_pronoun.png`. Stop-word rates are close. Those counts are the lexicon features fed to the Random Forest. `src/compute_linguistic_stats.py` writes the table without storing review text.

## What was not finished

The PDF lists next steps that were not closed in the local files: a side-by-side classifier for AI fakes vs Turker fakes, a tested prompt-feedback loop, and a human survey of perceived authenticity. Do not read the Ott-task accuracies as performance on GPT-4o-mini text.
