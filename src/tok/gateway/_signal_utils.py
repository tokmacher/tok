"""Shared signal utilities for gateway modules."""

from __future__ import annotations


def _merge_signal_counts(target: dict[str, int], extra: dict[str, int] | None) -> None:
    """Merge signal counts from extra into target in-place."""
    if not extra:
        return
    for key, value in extra.items():
        target[key] = target.get(key, 0) + value
