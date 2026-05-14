from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_detect_answer_phase import Step5Result, run_step_5
from tok.runtime.types import RuntimeRequest


def _make_request(**overrides) -> RuntimeRequest:
    defaults = dict(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hello"}],
        adapter_kind="unknown",
        tool_compatible=False,
    )
    defaults.update(overrides)
    return RuntimeRequest(**defaults)


class TestStep5ResultDefaults:
    def test_step5_result_has_correct_defaults(self) -> None:
        r = Step5Result()
        assert r.answer_ready is False
        assert r.late_answer_followthrough_active is False
        assert r.late_answer_assembly_repair_active is False
        assert r.late_answer_assembly_repair_mode == ""
        assert r.answer_ready_repair_active is False
        assert r.preserve_exact_search_evidence is False
        assert r.has_answer_anchor is False
        assert r.read_only_audit_turn is False
        assert r.tool_required_latch_active is False
        assert r.behavior_signals == {}
        assert r.runtime_hints == []
        assert r.resend_signals == {}
        assert r.exact_search_evidence_keys_in_request == set()

    def test_step5_result_all_fields_present(self) -> None:
        expected = {
            "answer_ready",
            "late_answer_followthrough_active",
            "late_answer_assembly_repair_active",
            "late_answer_assembly_repair_mode",
            "answer_ready_repair_active",
            "preserve_exact_search_evidence",
            "has_answer_anchor",
            "read_only_audit_turn",
            "tool_required_latch_active",
            "behavior_signals",
            "runtime_hints",
            "resend_signals",
            "exact_search_evidence_keys_in_request",
        }
        actual = {f.name for f in fields(Step5Result)}
        assert actual == expected


class TestStep5DetectAnswerPhase:
    def test_basic_request_no_tools_answer_ready_false(self) -> None:
        session = RuntimeSession()
        req = _make_request(tool_compatible=False)
        result = run_step_5(
            session=session,
            request=req,
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            normalized_tool_events=[],
            behavior_signals={},
            effective_tool_compatible=False,
            initial_answer_facts_present=False,
            initial_exact_search_evidence_present=False,
            exact_search_evidence_keys_in_request=set(),
            plan_finalization_turn=False,
            initial_runtime_hints=[],
        )
        assert result.answer_ready is False
        assert result.has_answer_anchor is False
        assert result.preserve_exact_search_evidence is False

    def test_answer_ready_flags_default_false_for_non_tool_compatible(self) -> None:
        session = RuntimeSession()
        req = _make_request(tool_compatible=False)
        result = run_step_5(
            session=session,
            request=req,
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            normalized_tool_events=[],
            behavior_signals={},
            effective_tool_compatible=False,
            initial_answer_facts_present=False,
            initial_exact_search_evidence_present=False,
            exact_search_evidence_keys_in_request=set(),
            plan_finalization_turn=False,
            initial_runtime_hints=[],
        )
        assert result.answer_ready is False
        assert result.late_answer_followthrough_active is False
        assert result.late_answer_assembly_repair_active is False
        assert result.late_answer_assembly_repair_mode == ""
        assert result.answer_ready_repair_active is False
        assert result.tool_required_latch_active is False

    def test_read_only_audit_turn_detected(self) -> None:
        session = RuntimeSession()
        req = _make_request(tool_compatible=True)
        result = run_step_5(
            session=session,
            request=req,
            translated_messages=[
                {"role": "user", "content": "audit the codebase in read-only mode with no edits and no network"},
            ],
            id_to_context={},
            normalized_tool_events=[],
            behavior_signals={},
            effective_tool_compatible=True,
            initial_answer_facts_present=False,
            initial_exact_search_evidence_present=False,
            exact_search_evidence_keys_in_request=set(),
            plan_finalization_turn=False,
            initial_runtime_hints=[],
        )
        assert result.read_only_audit_turn is True

    def test_tool_required_latch_streak_management(self) -> None:
        session = RuntimeSession()
        assert session._tool_required_latch_streak == 0
        req = _make_request(tool_compatible=True)
        run_step_5(
            session=session,
            request=req,
            translated_messages=[
                {
                    "role": "user",
                    "content": "use the read-only tools first to gather fresh evidence before finalizing",
                },
            ],
            id_to_context={},
            normalized_tool_events=[],
            behavior_signals={},
            effective_tool_compatible=True,
            initial_answer_facts_present=False,
            initial_exact_search_evidence_present=False,
            exact_search_evidence_keys_in_request=set(),
            plan_finalization_turn=False,
            initial_runtime_hints=[],
        )
        assert session._tool_required_latch_streak == 1
        assert session._tool_required_latch_streak >= 1

    def test_tool_required_latch_resets_on_resolved(self) -> None:
        session = RuntimeSession()
        session._tool_required_latch_streak = 3
        req = _make_request(tool_compatible=False)
        run_step_5(
            session=session,
            request=req,
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            normalized_tool_events=[],
            behavior_signals={},
            effective_tool_compatible=False,
            initial_answer_facts_present=False,
            initial_exact_search_evidence_present=False,
            exact_search_evidence_keys_in_request=set(),
            plan_finalization_turn=False,
            initial_runtime_hints=[],
        )
        assert session._tool_required_latch_streak == 0

    def test_answer_phase_expected_set_on_session(self) -> None:
        session = RuntimeSession()
        req = _make_request(tool_compatible=False)
        run_step_5(
            session=session,
            request=req,
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            normalized_tool_events=[],
            behavior_signals={},
            effective_tool_compatible=False,
            initial_answer_facts_present=False,
            initial_exact_search_evidence_present=False,
            exact_search_evidence_keys_in_request=set(),
            plan_finalization_turn=False,
            initial_runtime_hints=[],
        )
        assert session._answer_phase_expected_this_turn is False
