"""Shared signal utilities for gateway modules."""

from __future__ import annotations

from typing import Any


def _merge_signal_counts(target: dict[str, int], extra: dict[str, Any] | None) -> None:
    """Merge signal counts from extra into target in-place."""
    if not extra:
        return
    for key, value in extra.items():
        if isinstance(value, bool):
            normalized_value = int(value)
        elif isinstance(value, int):
            normalized_value = value
        else:
            continue
        target[key] = target.get(key, 0) + normalized_value
