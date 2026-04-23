from __future__ import annotations

STOPWORDS = {"the", "and", "a"}


def remove_stopwords(tokens: list[str]) -> list[str]:
    # BUG: this keeps only stopwords instead of removing them.
    return [token for token in tokens if token in STOPWORDS]
