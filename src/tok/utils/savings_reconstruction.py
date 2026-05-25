"""Ledger-derived session and lifetime savings summaries.

Reconstructs totals from per-request SavingsEvent JSONL files rather than
from in-memory counters, enabling offline auditing and consistency checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tok.utils.savings_event import SavingsEvent, read_savings_events

logger = logging.getLogger(__name__)

_EVENTS_FILENAME = "savings_events.jsonl"

__all__ = [
    "SavingsSummary",
    "reconstruct_session_summary",
    "reconstruct_session_summary_from_file",
    "reconstruct_lifetime_summary",
]


@dataclass
class SavingsSummary:
    """Aggregated savings metrics reconstructed from a JSONL event stream."""

    calls: int = 0
    baseline_input_tokens: int = 0
    actual_input_tokens: int = 0
    input_tokens_saved: int = 0
    baseline_output_tokens: int = 0
    actual_output_tokens: int = 0
    output_tokens_saved: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    baseline_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    fallback_count: int = 0
    degraded_count: int = 0


def reconstruct_session_summary(events: list[SavingsEvent]) -> SavingsSummary:
    """Compute a SavingsSummary by summing all events from a session."""
    summary = SavingsSummary()
    for ev in events:
        summary.calls += 1
        summary.baseline_input_tokens += ev.baseline_input_tokens
        summary.actual_input_tokens += ev.actual_input_tokens
        summary.input_tokens_saved += ev.input_tokens_saved
        summary.baseline_output_tokens += ev.baseline_output_tokens
        summary.actual_output_tokens += ev.actual_output_tokens
        summary.output_tokens_saved += ev.output_tokens_saved
        summary.cache_read_tokens += ev.cache_read_tokens
        summary.cache_write_tokens += ev.cache_write_tokens
        summary.baseline_cost_usd += ev.baseline_cost_usd
        summary.actual_cost_usd += ev.actual_cost_usd
        summary.cost_saved_usd += ev.cost_saved_usd
        if ev.fallback:
            summary.fallback_count += 1
        if ev.degraded_to_baseline:
            summary.degraded_count += 1
    # Enforce non-negativity invariant
    summary.input_tokens_saved = max(0, summary.input_tokens_saved)
    summary.output_tokens_saved = max(0, summary.output_tokens_saved)
    return summary


def reconstruct_session_summary_from_file(path: Path) -> SavingsSummary:
    """Read a JSONL file and reconstruct the session summary.

    Corrupt lines and records with missing required fields are skipped.
    Returns a zero SavingsSummary if the file does not exist.
    """
    events = read_savings_events(path)
    return reconstruct_session_summary(events)


def reconstruct_lifetime_summary(sessions_root: Path) -> SavingsSummary:
    """Reconstruct lifetime totals by scanning all session subdirectories.

    Scans ``sessions_root`` for direct subdirectories that contain a
    ``savings_events.jsonl`` file.  Directories without an events file are
    silently skipped.
    """
    sessions_root = Path(sessions_root)
    lifetime = SavingsSummary()
    if not sessions_root.exists():
        return lifetime

    for candidate in sessions_root.iterdir():
        if not candidate.is_dir():
            continue
        events_file = candidate / _EVENTS_FILENAME
        if not events_file.exists():
            continue
        session_summary = reconstruct_session_summary_from_file(events_file)
        lifetime.calls += session_summary.calls
        lifetime.baseline_input_tokens += session_summary.baseline_input_tokens
        lifetime.actual_input_tokens += session_summary.actual_input_tokens
        lifetime.input_tokens_saved += session_summary.input_tokens_saved
        lifetime.baseline_output_tokens += session_summary.baseline_output_tokens
        lifetime.actual_output_tokens += session_summary.actual_output_tokens
        lifetime.output_tokens_saved += session_summary.output_tokens_saved
        lifetime.cache_read_tokens += session_summary.cache_read_tokens
        lifetime.cache_write_tokens += session_summary.cache_write_tokens
        lifetime.baseline_cost_usd += session_summary.baseline_cost_usd
        lifetime.actual_cost_usd += session_summary.actual_cost_usd
        lifetime.cost_saved_usd += session_summary.cost_saved_usd
        lifetime.fallback_count += session_summary.fallback_count
        lifetime.degraded_count += session_summary.degraded_count

    lifetime.input_tokens_saved = max(0, lifetime.input_tokens_saved)
    lifetime.output_tokens_saved = max(0, lifetime.output_tokens_saved)
    return lifetime
