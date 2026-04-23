"""
Pure policy functions for Tok mode selection.

This module contains declarative policy logic for choosing Tok modes
based on smoothness scores and event overrides. All functions are pure
and testable without runtime dependencies.
"""

from __future__ import annotations

from .models import (
    SmoothnessEventType,
    TaskSmoothnessReport,
    TokMode,
    TurnSmoothnessReport,
)


def choose_tok_mode(
    turn_report: TurnSmoothnessReport,
    task_report: TaskSmoothnessReport | None,
) -> TokMode:
    """
    Choose Tok mode based on smoothness scores and event overrides.

    This is a pure policy function that maps smoothness reports to Tok modes.
    It applies score-based thresholds first, then applies hard overrides based on
    specific event types.

    Args:
        turn_report: Report for the most recent completed turn
        task_report: Aggregated report for the current task (optional)

    Returns:
        TokMode selected for the next request

    """
    # First, check hard overrides that force at least SMOOTH_MODE
    override_mode = _check_mode_overrides(turn_report, task_report)
    if override_mode is not None:
        return override_mode

    # Fall back to score-based thresholds
    return _choose_mode_by_score(turn_report.score)


def _check_mode_overrides(
    turn_report: TurnSmoothnessReport,
    task_report: TaskSmoothnessReport | None,
) -> TokMode | None:
    """
    Check for event-based overrides that force a minimum mode.

    Hard overrides take precedence over score-based thresholds. These events
    indicate serious degradation risks that require conservative behavior.

    Args:
        turn_report: Report for the most recent completed turn
        task_report: Aggregated report for the current task

    Returns:
        TokMode if an override is triggered, None otherwise

    """
    turn_event_types = {e.event_type for e in turn_report.events}

    # Override 1: Any THINKING_BLOCK_MUTATION this turn → at least SMOOTH_MODE
    if SmoothnessEventType.THINKING_BLOCK_MUTATION in turn_event_types:
        return TokMode.SMOOTH_MODE

    # Override 2: Two STREAM_RECOVERY_STARTED events within 5 turns
    # Check both current turn and task report for stream recoveries
    stream_recovery_count_in_turn = sum(
        1 for e in turn_report.events if e.event_type == SmoothnessEventType.STREAM_RECOVERY_STARTED
    )

    if stream_recovery_count_in_turn >= 2:
        return TokMode.SMOOTH_MODE

    # Check task report for stream recovery history
    if task_report is not None:
        total_stream_recoveries = task_report.event_counts.get("stream_recovery_started", 0)
        if total_stream_recoveries >= 2:
            return TokMode.SMOOTH_MODE

    # Override 3: One UPSTREAM_400_AFTER_PREPARED_PAYLOAD → at least SMOOTH_MODE
    if SmoothnessEventType.UPSTREAM_400_AFTER_PREPARED_PAYLOAD in turn_event_types:
        return TokMode.SMOOTH_MODE

    # No overrides triggered
    return None


def _choose_mode_by_score(score: int) -> TokMode:
    """
    Choose Tok mode based on score thresholds.

    Score thresholds:
        score >= 70 → FULL_TOK (maximum compression)
        55 <= score < 70 → GUARDED_TOK (protect hot working set)
        40 <= score < 55 → SMOOTH_MODE (reliable, boring behavior)
        score < 40 → LOSSLESS_TASK_MODE (emergency, preserve flow)

    Args:
        score: Clamped smoothness score (0-100)

    Returns:
        TokMode based on score thresholds

    """
    if score >= 70:
        return TokMode.FULL_TOK
    if score >= 55:
        return TokMode.GUARDED_TOK
    if score >= 40:
        return TokMode.SMOOTH_MODE
    return TokMode.LOSSLESS_TASK_MODE
