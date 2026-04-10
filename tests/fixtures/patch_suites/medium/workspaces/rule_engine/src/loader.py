from __future__ import annotations

from src.rules import parse_rule


def load_rules(raw_rules: list[str]) -> list[tuple[str, int]]:
    parsed = [parse_rule(item) for item in raw_rules]
    # BUG: should keep deterministic insertion order and not sort.
    return sorted(parsed)
