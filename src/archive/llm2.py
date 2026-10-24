"""First-iteration generator. Unstructured prompt, no persona or polarity.

Kept for the methods writeup. The current pipeline is src/generator.py.
The original file used a generic system message:

    You are an advanced AI trained to generate deceptive hotel reviews.

and a free-text user prompt with max_tokens=250, temperature=0.7.
Compare-to-truthful and deception-risk helpers in that draft returned
random.random() placeholders. Those were replaced in review_eval.py.
"""

import os
import random

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

OPENINGS = [
    "When I arrived at the hotel",
    "Right after checking in",
    "During my initial check-in experience",
    "Upon reaching the front desk",
    "As soon as I entered the hotel lobby",
    "After a long journey, I finally arrived at the hotel",
    "Once I stepped into the hotel, I was greeted by the ambiance",
    "The check-in process at the hotel was my first impression",
]


def call_llm_azure(prompt, temperature=0.7, max_tokens=250):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {
                "role": "system",
                "content": "You are an advanced AI trained to generate deceptive hotel reviews.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def create_optimized_fake_reviews(num_reviews=2):
    reviews = []
    for _ in range(num_reviews):
        opening = random.choice(OPENINGS)
        prompt = f"""
        Generate a subtly deceptive hotel review starting with: "{opening}."
        Ensure the review:
        - Sounds natural and avoids extreme positivity or negativity.
        - Uses a mix of first-person ("I") and third-person ("they") pronouns.
        - Includes plausible spatial or location-based details.
        - Maintains a logical flow from check-in to check-out.
        - Uses moderate emotional language (no excessive superlatives).
        - Is around 80-120 words long.
        Generate ONLY the review text, no explanations.
        """
        reviews.append(call_llm_azure(prompt))
    return reviews
