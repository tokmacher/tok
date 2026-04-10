from __future__ import annotations


def evaluate(record: dict[str, int], rules: list[tuple[str, int]]) -> bool:
    for key, minimum in rules:
        value = int(record.get(key, 0))
        # BUG: should be value >= minimum.
        if value <= minimum:
            return False
    return True
