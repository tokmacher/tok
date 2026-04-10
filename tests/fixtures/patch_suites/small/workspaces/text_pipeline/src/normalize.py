from __future__ import annotations


def normalize_text(text: str) -> str:
    # BUG: should lowercase and trim, but currently uppercases.
    return text.strip().upper()
