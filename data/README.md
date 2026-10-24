# Data

This repo does not ship the Deceptive Opinion Spam Corpus. Ott et al. released it under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Redistributing the 1,600 hotel reviews would violate that license.

## Ott corpus (required for training and EDA)

1. Request or download **Deceptive Opinion Spam Corpus v1.4** from [myleott.com/op-spam.html](https://myleott.com/op-spam.html).
2. Extract it so the folder looks like:

```
op_spam_v1.4/
  positive_polarity/
    deceptive_from_MTurk/fold1..fold5/*.txt
    truthful_from_TripAdvisor/fold1..fold5/*.txt
  negative_polarity/
    deceptive_from_MTurk/fold1..fold5/*.txt
    truthful_from_Web/fold1..fold5/*.txt
```

3. Point the loader at that folder:

```
export OP_SPAM_DIR=/path/to/op_spam_v1.4
```

or copy it to `data/op_spam_v1.4` (gitignored).

The corpus is 400 reviews in each of four cells: truthful/deceptive × positive/negative. Truthful positive reviews came from TripAdvisor. Truthful negative reviews came from Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor, and Yelp. Deceptive reviews were written by Mechanical Turk workers pretending to be guests of 20 Chicago hotels.

Cite Ott et al. 2011 and 2013 if you use the data.

## What is in this folder

`samples/` holds four GPT-4o-mini reviews produced by `src/generator.py` (two positive, two negative). They are examples of the generation pipeline, not evaluation data.

A Yelp polarity dump exists on the original workstation and is unused here. Do not commit it.
