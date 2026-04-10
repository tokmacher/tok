from __future__ import annotations

from src.normalize import normalize_text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    # BUG: should split on whitespace.
    return [part for part in normalized.split(",") if part]
