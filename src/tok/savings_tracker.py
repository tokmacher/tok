"""Backward-compatible facade for savings tracker helpers."""

from __future__ import annotations

from .utils.savings_tracker import (  # noqa: F401
    BASELINE_ONLY_SIGNAL,
    FALLBACK_SIGNAL,
    GLOBAL_LEDGER_FILENAME,
    SESSION_STATS_FILENAME,
    SavingsTracker,
    _default_ledger_path,
    _default_savings_file,
    _degradation_reason,
    _legacy_ledger_path,
    _session_quality,
)

__all__ = [
    "BASELINE_ONLY_SIGNAL",
    "FALLBACK_SIGNAL",
    "GLOBAL_LEDGER_FILENAME",
    "SESSION_STATS_FILENAME",
    "SavingsTracker",
    "_default_ledger_path",
    "_default_savings_file",
    "_degradation_reason",
    "_legacy_ledger_path",
    "_session_quality",
]
