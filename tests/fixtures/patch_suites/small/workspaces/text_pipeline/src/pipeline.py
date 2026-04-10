from __future__ import annotations

from src.tokenize import tokenize

STOPWORDS = {"the", "and", "a"}


def prepare_for_index(text: str) -> list[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    # BUG: should preserve original order while deduping.
    return sorted(set(tokens))
