"""Tests for request_preparation resend diagnostics - freshness signaling."""

from __future__ import annotations

import pytest

from tok.runtime.pipeline.request_preparation import (
    _annotate_full_turn_resend,
    _apply_tool_compatible_resend_diagnostics,
)


class TestApplyToolCompatibleResendDiagnostics:
    """Test suite for _apply_tool_compatible_resend_diagnostics function."""

    def test_suppressed_turn_uses_verified_current_terminology(self) -> None:
        """Verify suppressed state uses 'verified_current' not 'suppressed'."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {"state_resend_suppressed_turn": 1}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test memory",
            resend_signals,
            has_answer_anchor=False,
        )

        # Should use new terminology
        assert "state_resend_reason_state_verified_current" in behavior_signals
        # Old terminology should NOT appear
        assert "state_resend_reason_state_suppressed" not in behavior_signals

    def test_suppressed_with_answer_anchor(self) -> None:
        """Verify answer anchor with suppressed state uses new terminology."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {"state_resend_suppressed_turn": 1}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test memory",
            resend_signals,
            has_answer_anchor=True,
        )

        assert "answer_anchor_verified_current" in behavior_signals
        assert "answer_anchor_suppressed" not in behavior_signals

    def test_delta_turn_reports_delta_selected(self) -> None:
        """Verify delta resend reports correctly."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {"state_resend_delta_turn": 1}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test memory",
            resend_signals,
            has_answer_anchor=False,
        )

        assert behavior_signals.get("state_resend_reason_delta_selected") == 1

    def test_delta_with_answer_anchor(self) -> None:
        """Verify delta resend with answer anchor."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {"state_resend_delta_turn": 1}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test memory",
            resend_signals,
            has_answer_anchor=True,
        )

        assert behavior_signals.get("answer_anchor_delta_allowed") == 1

    def test_full_turn_with_answer_anchor(self) -> None:
        """Verify full resend with new answer anchor reason."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {"state_resend_full_turn": 1}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test memory",
            resend_signals,
            has_answer_anchor=True,
            resend_reason="new_answer_anchor",
        )

        assert behavior_signals.get("answer_anchor_forced_full_resend") == 1

    def test_payload_chars_recorded(self) -> None:
        """Verify payload size is recorded."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test memory payload",
            resend_signals,
            has_answer_anchor=False,
        )

        assert behavior_signals.get("state_payload_chars") == len("test memory payload")

    def test_answer_anchor_present_recorded(self) -> None:
        """Verify answer anchor presence is recorded."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {}

        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            "test",
            resend_signals,
            has_answer_anchor=True,
        )

        assert behavior_signals.get("answer_anchor_present") == 1


class TestAnnotateFullTurnResend:
    """Test suite for _annotate_full_turn_resend helper."""

    def test_new_answer_anchor_reason(self) -> None:
        """Verify new answer anchor reason is annotated."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {}

        _annotate_full_turn_resend(
            behavior_signals,
            resend_signals,
            resend_reason="new_answer_anchor",
            skip_reason_hint=None,
            tok_history_compression_skipped=False,
            tok_history_cut_point_missing=False,
            tool_compatible_compression=False,
        )

        assert behavior_signals.get("state_resend_reason_answer_anchor_present_kept_full") == 1

    def test_history_compression_skip_reason(self) -> None:
        """Verify history compression skip reason."""
        behavior_signals: dict[str, int] = {}
        resend_signals: dict[str, int] = {}

        _annotate_full_turn_resend(
            behavior_signals,
            resend_signals,
            resend_reason=None,
            skip_reason_hint="memory",
            tok_history_compression_skipped=True,
            tok_history_cut_point_missing=False,
            tool_compatible_compression=False,
        )

        assert behavior_signals.get("state_resend_reason_history_compression_skipped") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
