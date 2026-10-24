"""Generation-quality checks used while producing AI fake reviews.

The logistic regression here is a filter, not the reported detector.
It is fit on the full Ott corpus (no held-out split) so that newly
generated text can be scored for "looks like Mechanical Turk deception."
Reported classifier numbers live in results/metrics.json and come from
the bag-of-words and BERT notebooks.
"""

from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_dataset
from generator import generate_fake_reviews

_dataset = None
_truthful_reviews = None
_vectorizer_cls = None
_clf = None
_vectorizer_sim = None


def _ensure_models():
    global _dataset, _truthful_reviews, _vectorizer_cls, _clf, _vectorizer_sim
    if _clf is not None:
        return
    _dataset = load_dataset()
    _truthful_reviews = [
        r["review"] for r in _dataset if r["truthfulness"] == "truthful"
    ]
    labels = [1 if r["truthfulness"] == "deceptive" else 0 for r in _dataset]
    _vectorizer_cls = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
    X_cls = _vectorizer_cls.fit_transform([r["review"] for r in _dataset])
    _clf = LogisticRegression(random_state=42, max_iter=1000)
    _clf.fit(X_cls, labels)
    _vectorizer_sim = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    _vectorizer_sim.fit(_truthful_reviews)


def compare_to_truthful_reviews(
    fake_reviews: List[str], truthful_reviews: List[str]
) -> float:
    """Mean of each fake review's max cosine similarity to the truthful set."""
    _ensure_models()
    real_tfidf = _vectorizer_sim.transform(truthful_reviews)
    fake_tfidf = _vectorizer_sim.transform(fake_reviews)
    max_sims = cosine_similarity(fake_tfidf, real_tfidf).max(axis=1)
    return float(max_sims.mean())


def run_deception_classifier(fake_reviews: List[str]) -> float:
    """Mean P(deceptive) under the full-corpus TF-IDF logistic regression."""
    _ensure_models()
    probs = _clf.predict_proba(_vectorizer_cls.transform(fake_reviews))[:, 1]
    return float(probs.mean())


def evaluate_reviews(
    fake_reviews: List[str],
    truthful_reviews: List[str],
    polarity: str = "positive",
    temperature: float = 0.7,
) -> List[str]:
    """Regenerate the batch if mean P(deceptive) is above 0.5."""
    similarity_score = compare_to_truthful_reviews(fake_reviews, truthful_reviews)
    detection_risk = run_deception_classifier(fake_reviews)
    print(f"Similarity to truthful reviews: {similarity_score:.2f}")
    print(f"Deception detection risk:      {detection_risk:.2f}")
    if detection_risk > 0.5:
        print("Draft looks too fake. Regenerating at higher temperature.")
        return generate_fake_reviews(
            n=len(fake_reviews),
            polarity=polarity,
            temperature=temperature + 0.1,
        )
    return fake_reviews
