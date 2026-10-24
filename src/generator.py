"""Persona- and sentiment-conditioned hotel review generation via Azure OpenAI."""

from __future__ import annotations

import os
import random
import textwrap

from dotenv import load_dotenv
from openai import AzureOpenAI

from config import PERSONAS

load_dotenv()

OPENINGS = [
    "When I arrived at the hotel",
    "Right after checking in",
    "Upon reaching the front desk",
    "As soon as I entered the hotel lobby",
    "After a long journey, I finally arrived at the hotel",
    "Once I stepped into the hotel, I was greeted by the ambiance",
]

TOP_POS_PHRASES = ["spacious room", "river view", "helpful staff"]
TOP_NEG_PHRASES = ["dated décor", "musty smell", "weak coffee"]


def _client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not (endpoint and key and version):
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and "
            "AZURE_OPENAI_API_VERSION in .env. See .env.example."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=version,
        api_key=key,
    )


def _compose_prompt(opening: str, polarity: str, persona: str, max_words: int) -> str:
    tone = (
        "overall positive tone" if polarity == "positive" else "overall negative tone"
    )
    seed_phrases = TOP_POS_PHRASES if polarity == "positive" else TOP_NEG_PHRASES
    phrase = random.choice(seed_phrases)
    return textwrap.dedent(
        f"""
        You are {PERSONAS[persona]}. Write a hotel review with {tone} that starts with:
        "{opening}."
        • Include the phrase "{phrase}".
        • Target length ≈ 90-{max_words} words (hard cap {max_words}).
        • Mix "I/we" with an occasional "they".
        • Mention at least one spatial detail (floor, view, lobby, etc.).
        • Use clear, descriptive language but no more than two exclamation marks;
          avoid profanity.
        Return only the review text.
        """
    )


def generate_fake_reviews(
    n: int = 200,
    polarity: str = "positive",
    temperature: float = 0.7,
    max_words: int = 110,
) -> list[str]:
    """Generate n reviews, each truncated to max_words."""
    client = _client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise RuntimeError("Set AZURE_OPENAI_DEPLOYMENT_NAME in .env.")

    reviews: list[str] = []
    for _ in range(n):
        opening = random.choice(OPENINGS)
        persona = random.choice(list(PERSONAS))
        prompt = _compose_prompt(opening, polarity, persona, max_words)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI generating realistic hotel reviews.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        words = raw.split()
        reviews.append(" ".join(words[:max_words]))
    return reviews
