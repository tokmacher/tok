"""
Tests for smoothness policy behavior in runtime context.

These tests verify the policy.py functions work correctly in a runtime
context, including integration with session state and streaming behavior.
"""

from tok.runtime.core import RuntimeSession
from tok.runtime.smoothness.models import SmoothnessEventType, TokMode
from tok.runtime.smoothness.policy import choose_tok_mode
from tok.runtime.smoothness.scoring import score_turn


class TestThresholdBands:
    """Tests for each score threshold band."""

    def test_full_tok_threshold(self) -> None:
        """Score >= 70 should select FULL_TOK mode."""
        report = score_turn("turn_1", "task_1", [])
        mode = choose_tok_mode(report, None)
        assert mode == TokMode.FULL_TOK

    def test_guarded_tok_threshold(self) -> None:
        """Score 55-69 should select GUARDED_TOK mode."""
        events = [make_event(SmoothnessEventType.STREAM_READ_ERROR) for _ in range(3)]
        report = score_turn("turn_1", "task_1", events)
        mode = choose_tok_mode(report, None)
        assert report.score == 64
        assert mode == TokMode.GUARDED_TOK

    def test_smooth_mode_threshold(self) -> None:
        """Score 40-54 should select SMOOTH_MODE."""
        events = [make_event(SmoothnessEventType.STREAM_READ_ERROR) for _ in range(4)]
        report = score_turn("turn_1", "task_1", events)
        mode = choose_tok_mode(report, None)
        assert report.score == 52
        assert mode == TokMode.SMOOTH_MODE

    def test_lossless_task_mode_threshold(self) -> None:
        """Score < 40 should select LOSSLESS_TASK_MODE."""
        events = [make_event(SmoothnessEventType.STREAM_READ_ERROR) for _ in range(6)]
        report = score_turn("turn_1", "task_1", events)
        mode = choose_tok_mode(report, None)
        assert report.score == 28
        assert mode == TokMode.LOSSLESS_TASK_MODE


class TestModeOverrides:
    """Tests for event-based mode overrides."""

    def test_thinking_block_mutation_override(self) -> None:
        """THINKING_BLOCK_MUTATION should force SMOOTH_MODE regardless of score."""
        events = [make_event(SmoothnessEventType.THINKING_BLOCK_MUTATION)]
        report = score_turn("turn_1", "task_1", events)
        mode = choose_tok_mode(report, None)
        assert report.score == 65
        assert mode == TokMode.SMOOTH_MODE

    def test_repeated_stream_recovery_in_turn(self) -> None:
        """Two STREAM_RECOVERY_STARTED in one turn should force SMOOTH_MODE."""
        events = [
            make_event(SmoothnessEventType.STREAM_RECOVERY_STARTED),
            make_event(SmoothnessEventType.STREAM_RECOVERY_STARTED),
        ]
        report = score_turn("turn_1", "task_1", events)
        mode = choose_tok_mode(report, None)
        assert report.score == 80
        assert mode == TokMode.SMOOTH_MODE


class TestSmoothModeStreamingIntegration:
    """Integration test confirming SMOOTH_MODE affects streaming behavior."""

    def test_smooth_mode_disables_streaming_in_session(self) -> None:
        """SMOOTH_MODE should update session to disable streaming."""
        session = RuntimeSession()

        session.update_smoothness_state(
            turn_score=52,
            labour_index=1,
            tok_mode=TokMode.SMOOTH_MODE,
            event_counts={"stream_read_error": 4},
        )

        assert session.current_tok_mode == TokMode.SMOOTH_MODE

        should_stream = _should_enable_streaming(session)
        assert should_stream is False

    def test_full_tok_enables_streaming(self) -> None:
        """FULL_TOK mode should allow streaming."""
        session = RuntimeSession()

        session.update_smoothness_state(
            turn_score=100,
            labour_index=0,
            tok_mode=TokMode.FULL_TOK,
            event_counts={},
        )

        should_stream = _should_enable_streaming(session)
        assert should_stream is True


def make_event(event_type: SmoothnessEventType):
    """Helper to create a smoothness event with default penalty."""
    from tok.runtime.smoothness.models import SmoothnessEvent

    penalties = {
        SmoothnessEventType.STREAM_READ_ERROR: 12,
        SmoothnessEventType.STREAM_RECOVERY_STARTED: 10,
        SmoothnessEventType.THINKING_BLOCK_MUTATION: 35,
        SmoothnessEventType.REPEATED_ACTIVE_FILE_READ: 6,
        SmoothnessEventType.UPSTREAM_400_AFTER_PREPARED_PAYLOAD: 25,
        SmoothnessEventType.DIRECT_ACTION_AFTER_FIRST_READ: 0,
        SmoothnessEventType.USER_INTERRUPT_REDIRECTION: 12,
    }
    return SmoothnessEvent(
        event_type=event_type,
        turn_id="turn_1",
        task_id="task_1",
        penalty=penalties.get(event_type, 0),
    )


def _should_enable_streaming(session: RuntimeSession) -> bool:
    """Check if streaming should be enabled based on session mode."""
    return session.current_tok_mode not in (
        TokMode.SMOOTH_MODE,
        TokMode.LOSSLESS_TASK_MODE,
    )
