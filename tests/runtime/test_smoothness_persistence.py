"""
Tests for smoothness data persistence in session state.

These tests verify that smoothness reports flow correctly through
RuntimeSession and are accessible for policy decisions.
"""

from tok.runtime.core import RuntimeSession
from tok.runtime.smoothness.models import SmoothnessEventType, TokMode
from tok.runtime.smoothness.tracker import SmoothnessTracker


def test_finished_turn_updates_session_state() -> None:
    """Confirm finish_turn() updates RuntimeSession smoothness fields."""
    session = RuntimeSession()
    tracker = SmoothnessTracker()

    tracker.start_turn("turn_1", "task_1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    turn_report = tracker.finish_turn()

    # Build event counts from the report
    event_counts: dict[str, int] = {}
    for event in turn_report.events:
        key = event.event_type.value
        event_counts[key] = event_counts.get(key, 0) + 1

    # Update session state
    session.update_smoothness_state(
        turn_score=turn_report.score,
        labour_index=turn_report.labour_index,
        tok_mode=turn_report.mode,
        event_counts=event_counts,
    )

    # Verify session state was updated
    assert session.latest_turn_smoothness_score == 88
    assert session.latest_turn_labour_index == 0
    assert session.current_tok_mode == TokMode.FULL_TOK
    assert session.smoothness_event_counts.get("stream_read_error") == 1


def test_current_task_score_updates_across_turns() -> None:
    """Confirm task report updates across multiple turns."""
    session = RuntimeSession()
    tracker = SmoothnessTracker()

    # Turn 1: Stream read error
    tracker.start_turn("turn_1", "task_1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    report1 = tracker.finish_turn()

    event_counts1: dict[str, int] = {}
    for event in report1.events:
        key = event.event_type.value
        event_counts1[key] = event_counts1.get(key, 0) + 1

    session.update_smoothness_state(
        turn_score=report1.score,
        labour_index=report1.labour_index,
        tok_mode=report1.mode,
        event_counts=event_counts1,
    )

    # Verify first turn
    assert session.latest_turn_smoothness_score == 88
    assert session.current_task_smoothness_score == 88

    # Turn 2: No events (clean turn)
    tracker.start_turn("turn_2", "task_1")
    report2 = tracker.finish_turn()

    event_counts2: dict[str, int] = {}
    for event in report2.events:
        key = event.event_type.value
        event_counts2[key] = event_counts2.get(key, 0) + 1

    session.update_smoothness_state(
        turn_score=report2.score,
        labour_index=report2.labour_index,
        tok_mode=report2.mode,
        event_counts=event_counts2,
    )

    # Verify second turn updated state
    assert session.latest_turn_smoothness_score == 100
    assert session.current_task_smoothness_score == 100

    # Verify event counts accumulated
    assert session.smoothness_event_counts.get("stream_read_error") == 1


def test_labour_index_updates_correctly() -> None:
    """Confirm labour index is tracked correctly in session state."""
    session = RuntimeSession()
    tracker = SmoothnessTracker()

    tracker.start_turn("turn_1", "task_1")
    tracker.record(SmoothnessEventType.THINKING_BLOCK_MUTATION)
    tracker.record(SmoothnessEventType.REPEATED_ACTIVE_FILE_READ)
    turn_report = tracker.finish_turn()

    event_counts: dict[str, int] = {}
    for event in turn_report.events:
        key = event.event_type.value
        event_counts[key] = event_counts.get(key, 0) + 1

    session.update_smoothness_state(
        turn_score=turn_report.score,
        labour_index=turn_report.labour_index,
        tok_mode=turn_report.mode,
        event_counts=event_counts,
    )

    # Thinking mutation (2) + repeated read (1) = 3
    assert session.latest_turn_labour_index == 3


def test_tok_mode_updates_correctly() -> None:
    """Confirm Tok mode is tracked correctly in session state."""
    session = RuntimeSession()
    tracker = SmoothnessTracker()

    # Turn with very low score should trigger LOSSLESS_TASK_MODE
    # 6 STREAM_READ_ERROR events = 6 * 12 = 72 penalty
    # 100 - 72 = 28 score, which is < 40 threshold
    tracker.start_turn("turn_1", "task_1")
    for _ in range(6):
        tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    turn_report = tracker.finish_turn()

    event_counts: dict[str, int] = {}
    for event in turn_report.events:
        key = event.event_type.value
        event_counts[key] = event_counts.get(key, 0) + 1

    session.update_smoothness_state(
        turn_score=turn_report.score,
        labour_index=turn_report.labour_index,
        tok_mode=turn_report.mode,
        event_counts=event_counts,
    )

    assert session.current_tok_mode == TokMode.LOSSLESS_TASK_MODE
