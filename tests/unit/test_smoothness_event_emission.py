"""
Tests verifying smoothness events are emitted at the correct code sites.

These tests exercise the tracker directly to confirm events produce
the expected scores without requiring a live bridge session.
"""

from tok.runtime.smoothness.models import SmoothnessEventType, TokMode
from tok.runtime.smoothness.tracker import SmoothnessTracker


def test_stream_read_error_event() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR, {"error": "conn reset"})
    report = tracker.finish_turn()
    assert report.score == 88
    assert len(report.events) == 1
    assert report.events[0].event_type == SmoothnessEventType.STREAM_READ_ERROR


def test_thinking_block_mutation_event() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.THINKING_BLOCK_MUTATION)
    report = tracker.finish_turn()
    assert report.score == 65
    assert report.labour_index == 2
    assert report.mode == TokMode.SMOOTH_MODE


def test_repeated_active_file_read_event() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.REPEATED_ACTIVE_FILE_READ, {"count": 3})
    report = tracker.finish_turn()
    assert report.score == 94
    assert report.labour_index == 1
    assert report.mode == TokMode.FULL_TOK


def test_stream_recovery_started_event() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.STREAM_RECOVERY_STARTED)
    report = tracker.finish_turn()
    assert report.score == 90
    assert report.labour_index == 1


def test_stream_recovery_loop_breaker_event() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.STREAM_RECOVERY_LOOP_BREAKER)
    report = tracker.finish_turn()
    assert report.score == 92


def test_empty_stream_success_event() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.EMPTY_STREAM_SUCCESS)
    report = tracker.finish_turn()
    assert report.score == 92


def test_messages_changed_open_tool_loop() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.MESSAGES_CHANGED_OPEN_TOOL_LOOP)
    report = tracker.finish_turn()
    assert report.score == 88


def test_history_winnowing_active_loop() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.HISTORY_WINNOWING_ACTIVE_LOOP)
    report = tracker.finish_turn()
    assert report.score == 88


def test_prompt_optimization_active_task() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.PROMPT_OPTIMIZATION_ACTIVE_TASK)
    report = tracker.finish_turn()
    assert report.score == 94


def test_multiple_events_produce_correct_score() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    tracker.record(SmoothnessEventType.EMPTY_STREAM_SUCCESS)
    tracker.record(SmoothnessEventType.REPEATED_ACTIVE_FILE_READ)
    report = tracker.finish_turn()
    assert report.score == 100 - 12 - 8 - 6  # 74
    assert report.labour_index == 1


def test_task_report_aggregates_events() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.STREAM_READ_ERROR)
    tracker.finish_turn()

    tracker.start_turn("t2", "task1")
    tracker.record(SmoothnessEventType.THINKING_BLOCK_MUTATION)
    tracker.finish_turn()

    task = tracker.current_task_report()
    assert task is not None
    assert task.turn_count == 2
    assert task.worst_turn_score == 65
    assert task.event_counts.get("stream_read_error") == 1
    assert task.event_counts.get("thinking_block_mutation") == 1


def test_upstream_400_forces_smooth_mode() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.UPSTREAM_400_AFTER_PREPARED_PAYLOAD)
    report = tracker.finish_turn()
    assert report.mode == TokMode.SMOOTH_MODE
    assert report.score == 75


def test_two_stream_recoveries_force_smooth_mode() -> None:
    tracker = SmoothnessTracker()
    tracker.start_turn("t1", "task1")
    tracker.record(SmoothnessEventType.STREAM_RECOVERY_STARTED)
    tracker.record(SmoothnessEventType.STREAM_RECOVERY_STARTED)
    report = tracker.finish_turn()
    assert report.mode == TokMode.SMOOTH_MODE
    assert report.score == 80
