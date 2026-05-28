"""Text cleaning utilities for the TF-IDF pipeline."""

import re

STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "to", "of",
    "in", "on", "for", "by", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "has", "have", "had", "do", "does", "did", "not",
    "no", "yes", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "also", "as", "at", "from", "up", "down",
    "he", "she", "it", "we", "they", "you", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their"
])


def clean_text(text: str) -> str:
    """Return a normalized text string for TF-IDF vectorization."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [word for word in text.split() if word and word not in STOP_WORDS]
    return " ".join(words)


def clean_documents(texts):
    return [clean_text(text) for text in texts]
   