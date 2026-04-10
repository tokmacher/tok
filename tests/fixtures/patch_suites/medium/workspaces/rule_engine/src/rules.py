from __future__ import annotations


def parse_rule(raw: str) -> tuple[str, int]:
    # format: key>=value
    key, value = raw.split(">=")
    # BUG: key should be stripped and lowercased.
    return key, int(value)
