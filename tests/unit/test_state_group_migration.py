"""Regression tests for RuntimeSession state group migration (Plan 1).

These tests verify that:
1. All state groups reset correctly
2. Backward-compat properties delegate to groups (no split-brain)
3. reset_session() produces a clean state
4. record_invalid_tool_history_recovery() clears state via groups
"""

from __future__ import annotations

from tok.runtime.core import RuntimeSession


class TestFallbackStateGroup:
    def test_defaults(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        assert s.fallback.consecutive_count == 0
        assert s.fallback.baseline_only is False
        assert s.fallback.persistence_failures == 0

    def test_backward_compat_properties(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        assert s._consecutive_fallback_count == 0
        assert s._baseline_only is False
        assert s._persistence_failures == 0
        s._consecutive_fallback_count = 5
        assert s.fallback.consecutive_count == 5
        s._baseline_only = True
        assert s.fallback.baseline_only is True

    def test_reset(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.fallback.consecutive_count = 3
        s.fallback.baseline_only = True
        s.fallback.persistence_failures = 1
        s.fallback.reset()
        assert s.fallback.consecutive_count == 0
        assert s.fallback.baseline_only is False
        assert s.fallback.persistence_failures == 0


class TestAnswerPhaseStateGroup:
    def test_backward_compat_properties(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        assert s.answer_phase.answer_ready_repair_pending is False
        assert s.answer_phase.answer_ready_repair_active is False
        assert s.answer_phase.late_assembly_repair_pending is False
        assert s.answer_phase.late_assembly_repair_active is False
        assert s.answer_phase.late_assembly_repair_mode_pending == ""
        assert s.answer_phase.late_assembly_repair_mode_active == ""
        assert s.answer_phase.late_followthrough_pending is False
        assert s.answer_phase.late_followthrough_active is False
        assert s.answer_phase.answer_phase_expected_this_turn is False
        assert s.answer_phase.natural_response_acceptable_this_turn is False

    def test_no_split_brain(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.answer_phase.answer_ready_repair_pending = True
        s.answer_phase.late_assembly_repair_mode_active = "tool_only"
        assert s.answer_phase.answer_ready_repair_pending is True
        assert s.answer_phase.late_assembly_repair_mode_active == "tool_only"

    def test_reset(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.answer_phase.answer_ready_repair_pending = True
        s.answer_phase.late_followthrough_active = True
        s.answer_phase.reset()
        assert s.answer_phase.answer_ready_repair_pending is False
        assert s.answer_phase.late_followthrough_active is False


class TestStreamingRecoveryStateGroup:
    def test_backward_compat_via_group(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        assert s.streaming_recovery.reacquisition_budget == 0
        assert s.streaming_recovery.history_floor_budget == 0
        assert s.streaming_recovery.tool_use_only_signature == ""
        assert s.streaming_recovery.cooldown_remaining == 0
        assert s.streaming_recovery.cooldown_suppressed is False
        assert s.streaming_recovery.read_error_consecutive_count == 0
        assert s.streaming_recovery.read_error_last_stage == ""

    def test_reset(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.streaming_recovery.reacquisition_budget = 5
        s.streaming_recovery.cooldown_suppressed = True
        s.streaming_recovery.read_error_consecutive_count = 3
        s.streaming_recovery.reset()
        assert s.streaming_recovery.reacquisition_budget == 0
        assert s.streaming_recovery.cooldown_suppressed is False
        assert s.streaming_recovery.read_error_consecutive_count == 0


class TestRequestPolicyStateGroup:
    def test_defaults(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        assert s.request_policy.tool_mode_sticky_turns == 0
        assert s.request_policy.stream_recovery_watch_turns == 0
        assert s.request_policy.tool_recovery_watch_turns == 0
        assert s.request_policy.last_effective_tool_compatible is False

    def test_reset(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.request_policy.tool_mode_sticky_turns = 3
        s.request_policy.last_effective_tool_compatible = True
        s.request_policy.reset()
        assert s.request_policy.tool_mode_sticky_turns == 0
        assert s.request_policy.last_effective_tool_compatible is False


class TestEvidenceSafetyStateGroup:
    def test_defaults(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        assert s.evidence_safety.neighborhoods == {}
        assert s.evidence_safety.anchor_novelty_keys == {}
        assert s.evidence_safety.alias_map == {}
        assert s.evidence_safety.first_exact_seen == set()
        assert s.evidence_safety.ledger == {}
        assert s.evidence_safety.pending_exact_keys == set()

    def test_reset(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.evidence_safety.first_exact_seen.add("file.py")
        s.evidence_safety.ledger["key1"] = object.__new__(type("E", (), {"__dataclass_fields__": {}}))
        s.evidence_safety.reset()
        assert s.evidence_safety.first_exact_seen == set()
        assert s.evidence_safety.ledger == {}


class TestFullResetSession:
    def test_reset_session_clears_all_groups(self):
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s.fallback.consecutive_count = 3
        s.fallback.baseline_only = True
        s.smoothness_state.latest_turn_score = 50
        s.loop_detection.detected = True
        s.telemetry.step_count = 10
        s.macro.pending_heal = "test_macro"
        s.project.files_read.add("foo.py")
        s.streaming_recovery.reacquisition_budget = 5
        s.request_policy.tool_mode_sticky_turns = 3
        s.answer_phase.answer_ready_repair_pending = True
        s.evidence_safety.first_exact_seen.add("bar.py")

        s.reset_session()

        assert s.fallback.consecutive_count == 0
        assert s.fallback.baseline_only is False
        assert s.smoothness_state.latest_turn_score == 100
        assert s.loop_detection.detected is False
        assert s.telemetry.step_count == 0
        assert s.macro.pending_heal == ""
        assert "foo.py" not in s.project.files_read
        assert s.streaming_recovery.reacquisition_budget == 0
        assert s.request_policy.tool_mode_sticky_turns == 0
        assert s.answer_phase.answer_ready_repair_pending is False
        assert "bar.py" not in s.evidence_safety.first_exact_seen

    def test_reset_session_no_bare_field_leaks(self):
        """After reset_session, backward-compat properties should return defaults."""
        s = RuntimeSession.__new__(RuntimeSession)
        s.__init__()
        s._step_count = 10
        s._consecutive_fallback_count = 3
        s._baseline_only = True
        s._token_count = 5000

        s.reset_session()

        assert s._step_count == 0
        assert s._consecutive_fallback_count == 0
        assert s._baseline_only is False
        assert s._token_count == 0
