"""Savings tracking and ledger I/O."""

from __future__ import annotations

__all__ = [
    "BASELINE_ONLY_SIGNAL",
    "FALLBACK_SIGNAL",
    "GLOBAL_LEDGER_FILENAME",
    "SESSION_STATS_FILENAME",
    "SavingsTracker",
]

import logging

from .utils.savings_tracker import (  # noqa: F401 — re-export for backward compat
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

logger = logging.getLogger("tok.stats")
