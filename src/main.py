"""Generate a balanced batch of AI hotel reviews and write them to disk."""

from pathlib import Path

from data_loader import load_dataset
from generator import generate_fake_reviews
from review_eval import evaluate_reviews

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "llm_generated"  # overwrites the committed reviews
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_individual_files(reviews: list[str], polarity: str) -> None:
    subdir = OUT_DIR / polarity
    subdir.mkdir(parents=True, exist_ok=True)
    for i, review in enumerate(reviews, 1):
        path = subdir / f"{polarity}_{i:02d}.txt"
        path.write_text(review + "\n", encoding="utf-8")


def gen_refine_save(polarity: str, n: int = 200) -> None:
    truthful_texts = [
        r["review"] for r in load_dataset() if r["truthfulness"] == "truthful"
    ]
    raw_batch = generate_fake_reviews(
        n=n, polarity=polarity, temperature=0.7, max_words=110
    )
    refined = evaluate_reviews(
        raw_batch, truthful_texts, polarity=polarity, temperature=0.7
    )
    save_individual_files(refined, polarity)
    print(f"Saved {len(refined)} {polarity} reviews to {OUT_DIR / polarity}")


def main() -> None:
    for polarity in ("positive", "negative"):
        gen_refine_save(polarity)


if __name__ == "__main__":
    main()
