"""
Tests for smoothness scoring model.

These tests verify the pure scoring functions and penalty calculations
without any runtime wiring.
"""

import pytest

from tok.runtime.smoothness.models import (
    SmoothnessEvent,
    SmoothnessEventType,
    TokMode,
)
from tok.runtime.smoothness.policy import choose_tok_mode
from tok.runtime.smoothness.scoring import score_task, score_turn
from tok.runtime.smoothness.tracker import SmoothnessTracker


def test_clean_turn_no_events() -> None:
    """A turn with no events should score 100."""
    report = score_turn("turn_1", "task_1", [])
    assert report.score == 100
    assert report.labour_index == 0
    assert report.mode == TokMode.FULL_TOK


def test_one_stream_read_error() -> None:
    """A turn with one STREAM_READ_ERROR should score 88 (100 - 12)."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_READ_ERROR,
            turn_id="turn_1",
            task_id="task_1",
            penalty=12,
        )
    ]
    report = score_turn("turn_1", "task_1", events)
    assert report.score == 88
    assert report.labour_index == 0


def test_one_thinking_block_mutation() -> None:
    """A turn with one THINKING_BLOCK_MUTATION should score 65 (100 - 35)."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.THINKING_BLOCK_MUTATION,
            turn_id="turn_1",
            task_id="task_1",
            penalty=35,
        )
    ]
    report = score_turn("turn_1", "task_1", events)
    assert report.score == 65
    assert report.labour_index == 2


def test_clamping_below_zero() -> None:
    """Score should clamp to 0 even with many penalties."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.THINKING_BLOCK_MUTATION,
            turn_id="turn_1",
            task_id="task_1",
            penalty=35,
        )
        for _ in range(5)
    ]
    report = score_turn("turn_1", "task_1", events)
    assert report.score == 0


def test_score_task_aggregation() -> None:
    """Task score should average across turns."""
    turn1 = score_turn(
        "turn_1",
        "task_1",
        [
            SmoothnessEvent(
                event_type=SmoothnessEventType.STREAM_READ_ERROR,
                turn_id="turn_1",
                task_id="task_1",
                penalty=12,
            )
        ],
    )
    turn2 = score_turn("turn_2", "task_1", [])

    task_report = score_task("task_1", [turn1, turn2])

    assert task_report.average_turn_score == 94.0
    assert task_report.worst_turn_score == 88
    assert task_report.task_score == 94
    assert task_report.turn_count == 2


def test_task_event_counts() -> None:
    """Task report should aggregate event counts."""
    turn1 = score_turn(
        "turn_1",
        "task_1",
        [
            SmoothnessEvent(
                event_type=SmoothnessEventType.STREAM_READ_ERROR,
                turn_id="turn_1",
                task_id="task_1",
                penalty=12,
            )
        ],
    )
    turn2 = score_turn(
        "turn_2",
        "task_1",
        [
            SmoothnessEvent(
                event_type=SmoothnessEventType.STREAM_READ_ERROR,
                turn_id="turn_2",
                task_id="task_1",
                penalty=12,
            ),
            SmoothnessEvent(
                event_type=SmoothnessEventType.REPEATED_ACTIVE_FILE_READ,
                turn_id="turn_2",
                task_id="task_1",
                penalty=6,
            ),
        ],
    )

    task_report = score_task("task_1", [turn1, turn2])

    assert task_report.event_counts.get("stream_read_error") == 2
    assert task_report.event_counts.get("repeated_active_file_read") == 1


def test_mode_full_tok() -> None:
    """Score >= 70 should result in FULL_TOK mode."""
    report = score_turn("turn_1", "task_1", [])
    mode = choose_tok_mode(report, None)
    assert mode == TokMode.FULL_TOK


def test_mode_guarded_tok() -> None:
    """Score 55-69 should result in GUARDED_TOK mode."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_READ_ERROR,
            turn_id="turn_1",
            task_id="task_1",
            penalty=12,
        )
        for _ in range(3)
    ]
    report = score_turn("turn_1", "task_1", events)
    mode = choose_tok_mode(report, None)
    assert report.score == 64
    assert mode == TokMode.GUARDED_TOK


def test_mode_smooth_mode_by_score() -> None:
    """Score 40-54 should result in SMOOTH_MODE."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_READ_ERROR,
            turn_id="turn_1",
            task_id="task_1",
            penalty=12,
        )
        for _ in range(4)
    ]
    report = score_turn("turn_1", "task_1", events)
    mode = choose_tok_mode(report, None)
    assert report.score == 52
    assert mode == TokMode.SMOOTH_MODE


def test_mode_lossless_by_score() -> None:
    """Score < 40 should result in LOSSLESS_TASK_MODE."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_READ_ERROR,
            turn_id="turn_1",
            task_id="task_1",
            penalty=12,
        )
        for _ in range(6)
    ]
    report = score_turn("turn_1", "task_1", events)
    mode = choose_tok_mode(report, None)
    assert report.score == 28
    assert mode == TokMode.LOSSLESS_TASK_MODE


def test_mode_override_thinking_mutation() -> None:
    """THINKING_BLOCK_MUTATION should force SMOOTH_MODE regardless of score."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.THINKING_BLOCK_MUTATION,
            turn_id="turn_1",
            task_id="task_1",
            penalty=35,
        )
    ]
    report = score_turn("turn_1", "task_1", events)
    mode = choose_tok_mode(report, None)
    assert report.score == 65
    assert mode == TokMode.SMOOTH_MODE


def test_mode_override_repeated_stream_recovery() -> None:
    """Two STREAM_RECOVERY_STARTED events should force SMOOTH_MODE."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_RECOVERY_STARTED,
            turn_id="turn_1",
            task_id="task_1",
            penalty=10,
        ),
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_RECOVERY_STARTED,
            turn_id="turn_1",
            task_id="task_1",
            penalty=10,
        ),
    ]
    report = score_turn("turn_1", "task_1", events)
    mode = choose_tok_mode(report, None)
    assert report.score == 80
    assert mode == TokMode.SMOOTH_MODE


def test_tracker_basic_workflow() -> None:
    """SmoothnessTracker should track events across turns."""
    tracker = SmoothnessTracker()

    tracker.start_turn("turn_1", "task_1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    report = tracker.finish_turn()

    assert report.turn_id == "turn_1"
    assert report.task_id == "task_1"
    assert report.score == 88
    assert len(report.events) == 1


def test_tracker_task_aggregation() -> None:
    """SmoothnessTracker should aggregate reports across turns."""
    tracker = SmoothnessTracker()

    tracker.start_turn("turn_1", "task_1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    tracker.finish_turn()

    tracker.start_turn("turn_2", "task_1")
    tracker.finish_turn()

    task_report = tracker.current_task_report()
    assert task_report is not None
    assert task_report.turn_count == 2
    assert task_report.average_turn_score == 94.0


def test_tracker_without_start_raises() -> None:
    """Calling record() before start_turn() should raise."""
    tracker = SmoothnessTracker()
    with pytest.raises(RuntimeError, match="Must call start_turn"):
        tracker.record(SmoothnessEventType.STREAM_READ_ERROR)


def test_tracker_finish_without_start_raises() -> None:
    """Calling finish_turn() before start_turn() should raise."""
    tracker = SmoothnessTracker()
    with pytest.raises(RuntimeError, match="Must call start_turn"):
        tracker.finish_turn()


def test_bonus_direct_action() -> None:
    """DIRECT_ACTION_AFTER_FIRST_READ should add 5 to score, but clamped to 100."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.DIRECT_ACTION_AFTER_FIRST_READ,
            turn_id="turn_1",
            task_id="task_1",
            penalty=0,
        )
    ]
    report = score_turn("turn_1", "task_1", events)
    assert report.score == 100  # Clamped from 105


def test_labour_index_calculation() -> None:
    """Labour index should weight thinking mutations double."""
    events = [
        SmoothnessEvent(
            event_type=SmoothnessEventType.STREAM_RECOVERY_STARTED,
            turn_id="turn_1",
            task_id="task_1",
            penalty=10,
        ),
        SmoothnessEvent(
            event_type=SmoothnessEventType.REPEATED_ACTIVE_FILE_READ,
            turn_id="turn_1",
            task_id="task_1",
            penalty=6,
        ),
        SmoothnessEvent(
            event_type=SmoothnessEventType.THINKING_BLOCK_MUTATION,
            turn_id="turn_1",
            task_id="task_1",
            penalty=35,
        ),
        SmoothnessEvent(
            event_type=SmoothnessEventType.USER_INTERRUPT_REDIRECTION,
            turn_id="turn_1",
            task_id="task_1",
            penalty=12,
        ),
    ]
    report = score_turn("turn_1", "task_1", events)
    assert report.labour_index == 5
