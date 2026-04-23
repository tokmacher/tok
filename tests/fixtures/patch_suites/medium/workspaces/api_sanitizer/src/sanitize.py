from __future__ import annotations


def sanitize_email(value: str) -> str:
    # BUG: should lowercase and trim surrounding whitespace.
    return value


def sanitize_username(value: str) -> str:
    cleaned = value.strip()
    # BUG: should collapse internal spaces to underscores.
    return cleaned
