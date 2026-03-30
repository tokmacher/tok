"""Savings tracking and ledger I/O."""

from __future__ import annotations

import logging

from ..savings_tracker import (  # noqa: F401 — re-export for backward compat
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
    "calculate_reasoning_depth_per_token",
]


def calculate_reasoning_depth_per_token(
    step_count: int, tool_diversity: int, token_count: int
) -> float:
    """Dual-axis metric: reasoning diversity per token consumed.

    step_count      — number of assistant response steps in the session
    tool_diversity  — number of distinct tool names used
    token_count     — total output tokens consumed

    Higher is better.  Returns 0.0 when token_count is 0.
    """
    if token_count == 0:
        return 0.0
    return round((step_count * max(tool_diversity, 1)) / token_count, 4)
