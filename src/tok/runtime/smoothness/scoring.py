"""
Pure scoring functions for smoothness calculation.

This module contains the penalty/bonus tables and pure functions for computing
smoothness scores. No runtime state or I/O here.
"""

from __future__ import annotations

from .models import (
    SmoothnessEvent,
    SmoothnessEventType,
    TaskSmoothnessReport,
    TokMode,
    TurnSmoothnessReport,
)

TURN_START_SCORE = 100

PENALTIES: dict[SmoothnessEventType, int] = {
    SmoothnessEventType.STREAM_READ_ERROR: 12,
    SmoothnessEventType.EMPTY_STREAM_SUCCESS: 8,
    SmoothnessEventType.STREAM_RECOVERY_STARTED: 10,
    SmoothnessEventType.STREAM_RECOVERY_LOOP_BREAKER: 8,
    SmoothnessEventType.UPSTREAM_400_AFTER_PREPARED_PAYLOAD: 25,
    SmoothnessEventType.THINKING_BLOCK_MUTATION: 35,
    SmoothnessEventType.MESSAGES_CHANGED_OPEN_TOOL_LOOP: 12,
    SmoothnessEventType.HISTORY_WINNOWING_ACTIVE_LOOP: 12,
    SmoothnessEventType.SEMANTIC_DEDUP_ACTIVE_FILE: 10,
    SmoothnessEventType.PROMPT_OPTIMIZATION_ACTIVE_TASK: 6,
    SmoothnessEventType.REPEATED_ACTIVE_FILE_READ: 6,
    SmoothnessEventType.REPEATED_SEARCH_SAME_TARGET: 4,
    SmoothnessEventType.USER_INTERRUPT_REDIRECTION: 12,
}

BONUSES: dict[SmoothnessEventType, int] = {
    SmoothnessEventType.DIRECT_ACTION_AFTER_FIRST_READ: 5,
}


def score_turn(
    turn_id: str,
    task_id: str,
    events: list[SmoothnessEvent],
) -> TurnSmoothnessReport:
    """
    Compute smoothness score for a single turn.

    Args:
        turn_id: Unique identifier for this turn
        task_id: Task this turn belongs to
        events: List of smoothness events that occurred

    Returns:
        TurnSmoothnessReport with score clamped to 0-100

    """
    score = TURN_START_SCORE

    for event in events:
        if event.event_type in PENALTIES:
            score -= PENALTIES[event.event_type]
        elif event.event_type in BONUSES:
            score += BONUSES[event.event_type]

    labour_index = _compute_labour_index(events)

    clamped_score = max(0, min(100, score))

    return TurnSmoothnessReport(
        turn_id=turn_id,
        task_id=task_id,
        score=clamped_score,
        labour_index=labour_index,
        mode=TokMode.FULL_TOK,  # Placeholder, will be set by tracker
        events=events,
    )


def score_task(
    task_id: str,
    turn_reports: list[TurnSmoothnessReport],
) -> TaskSmoothnessReport:
    """
    Aggregate smoothness scores across multiple turns in a task.

    Args:
        task_id: Unique identifier for this task
        turn_reports: List of turn reports to aggregate

    Returns:
        TaskSmoothnessReport with aggregated metrics

    """
    if not turn_reports:
        return TaskSmoothnessReport(
            task_id=task_id,
            average_turn_score=100.0,
            worst_turn_score=100,
            task_score=100,
            labour_index=0,
            event_counts={},
            turn_count=0,
        )

    scores = [r.score for r in turn_reports]
    average_turn_score = sum(scores) / len(scores)
    worst_turn_score = min(scores)

    task_score = int(average_turn_score)

    total_labour_index = sum(r.labour_index for r in turn_reports)

    event_counts: dict[str, int] = {}
    for report in turn_reports:
        for event in report.events:
            key = event.event_type.value
            event_counts[key] = event_counts.get(key, 0) + 1

    return TaskSmoothnessReport(
        task_id=task_id,
        average_turn_score=round(average_turn_score, 1),
        worst_turn_score=worst_turn_score,
        task_score=task_score,
        labour_index=total_labour_index,
        event_counts=event_counts,
        turn_count=len(turn_reports),
    )


def _compute_labour_index(events: list[SmoothnessEvent]) -> int:
    """
    Compute labour index for a turn.

    Labour index = stream recoveries + repeated reads + (thinking mutations * 2) + user interruptions
    """
    labour = 0

    for event in events:
        if event.event_type in (
            SmoothnessEventType.STREAM_RECOVERY_STARTED,
            SmoothnessEventType.REPEATED_ACTIVE_FILE_READ,
        ):
            labour += 1
        elif event.event_type == SmoothnessEventType.THINKING_BLOCK_MUTATION:
            labour += 2
        elif event.event_type == SmoothnessEventType.USER_INTERRUPT_REDIRECTION:
            labour += 1

    return labour
