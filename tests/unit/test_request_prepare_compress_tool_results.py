from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_compress_tool_results import (
    Step6Result,
    _first_exact_evidence_seen_for_compression,
    _retains_required_exact_search_evidence,
    run_step_6,
)
from tok.runtime.types import RuntimeRequest


class TestStep6ResultDefaults:
    def test_step6_result_has_correct_defaults(self) -> None:
        r = Step6Result()
        assert r.body == {}
        assert r.type_breakdown == {}
        assert r.saved_tokens == 0
        assert r.compressed is False
        assert r.current_path == ""
        assert r.behavior_signals == {}
        assert r.runtime_hints == []
        assert r.compress_tool_results_bypassed is False

    def test_step6_result_all_fields_present(self) -> None:
        expected = {
            "body",
            "type_breakdown",
            "saved_tokens",
            "compressed",
            "current_path",
            "behavior_signals",
            "runtime_hints",
            "compress_tool_results_bypassed",
        }
        actual = {f.name for f in fields(Step6Result)}
        assert actual == expected


class TestStep6CompressToolResults:
    def test_first_exact_evidence_seen_no_preserve(self) -> None:
        session = RuntimeSession()
        result = _first_exact_evidence_seen_for_compression(session, False, set())
        assert isinstance(result, set)

    def test_first_exact_evidence_seen_with_preserve(self) -> None:
        session = RuntimeSession()
        result = _first_exact_evidence_seen_for_compression(session, True, set())
        assert isinstance(result, set)

    def test_retains_required_exact_search_evidence_no_preserve(self) -> None:
        messages = [{"role": "user", "content": []}]
        id_to_context = {}
        result = _retains_required_exact_search_evidence(messages, id_to_context, False, {"search|foo"})
        assert result is True

    def test_retains_required_exact_search_evidence_missing_key(self) -> None:
        messages = [{"role": "user", "content": []}]
        id_to_context = {}
        result = _retains_required_exact_search_evidence(messages, id_to_context, True, {"search|foo"})
        assert result is False

    def test_compress_tool_results_not_bypassed(self) -> None:
        session = RuntimeSession()
        behavior_signals: dict[str, int] = {}
        request = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="unknown",
            tool_compatible=False,
        )
        result = run_step_6(
            session=session,
            request=request,
            body={"messages": [{"role": "user", "content": "hello"}]},
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            behavior_signals=behavior_signals,
            effective_tool_compatible=True,
            preserve_exact_search_evidence=False,
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            plan_finalization_turn=False,
            mode="balanced",
            policy=_MockPolicy(),
            exact_search_evidence_keys_in_request=set(),
            current_pressure=0,
            saved_tokens=0,
            compressed=False,
            result_cache=None,
        )
        assert result.compress_tool_results_bypassed is False

    def test_broad_audit_batch_bypasses_compression(self) -> None:
        session = RuntimeSession()
        behavior_signals: dict[str, int] = {}
        request = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="unknown",
            tool_compatible=False,
        )
        result = run_step_6(
            session=session,
            request=request,
            body={"messages": [{"role": "user", "content": "hello"}]},
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            behavior_signals=behavior_signals,
            effective_tool_compatible=True,
            preserve_exact_search_evidence=False,
            broad_audit_batch=True,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            plan_finalization_turn=False,
            mode="balanced",
            policy=_MockPolicy(),
            exact_search_evidence_keys_in_request=set(),
            current_pressure=0,
            saved_tokens=0,
            compressed=False,
            result_cache=None,
        )
        assert result.compress_tool_results_bypassed is True
        assert result.behavior_signals.get("broad_audit_tool_result_compression_skipped") == 1

    def test_edit_reacquisition_signals_bypasses_compression(self) -> None:
        session = RuntimeSession()
        behavior_signals: dict[str, int] = {}
        request = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="unknown",
            tool_compatible=False,
        )
        result = run_step_6(
            session=session,
            request=request,
            body={"messages": [{"role": "user", "content": "hello"}]},
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            behavior_signals=behavior_signals,
            effective_tool_compatible=True,
            preserve_exact_search_evidence=False,
            broad_audit_batch=False,
            edit_reacquisition_signals={"some_signal": 1},
            stream_recovery_history_floor_active=False,
            plan_finalization_turn=False,
            mode="balanced",
            policy=_MockPolicy(),
            exact_search_evidence_keys_in_request=set(),
            current_pressure=0,
            saved_tokens=0,
            compressed=False,
            result_cache=None,
        )
        assert result.compress_tool_results_bypassed is True
        assert result.behavior_signals.get("evidence_tool_result_compression_skipped") == 1

    def test_stream_recovery_history_floor_active_bypasses_compression(self) -> None:
        session = RuntimeSession()
        behavior_signals: dict[str, int] = {}
        request = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="unknown",
            tool_compatible=False,
        )
        result = run_step_6(
            session=session,
            request=request,
            body={"messages": [{"role": "user", "content": "hello"}]},
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            behavior_signals=behavior_signals,
            effective_tool_compatible=True,
            preserve_exact_search_evidence=False,
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=True,
            plan_finalization_turn=False,
            mode="balanced",
            policy=_MockPolicy(),
            exact_search_evidence_keys_in_request=set(),
            current_pressure=0,
            saved_tokens=0,
            compressed=False,
            result_cache=None,
        )
        assert result.compress_tool_results_bypassed is True

    def test_plan_finalization_turn_bypasses_compression(self) -> None:
        session = RuntimeSession()
        behavior_signals: dict[str, int] = {}
        request = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="unknown",
            tool_compatible=False,
        )
        result = run_step_6(
            session=session,
            request=request,
            body={"messages": [{"role": "user", "content": "hello"}]},
            translated_messages=[{"role": "user", "content": "hello"}],
            id_to_context={},
            behavior_signals=behavior_signals,
            effective_tool_compatible=True,
            preserve_exact_search_evidence=False,
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            plan_finalization_turn=True,
            mode="balanced",
            policy=_MockPolicy(),
            exact_search_evidence_keys_in_request=set(),
            current_pressure=0,
            saved_tokens=0,
            compressed=False,
            result_cache=None,
        )
        assert result.compress_tool_results_bypassed is True
        assert result.behavior_signals.get("plan_finalization_tool_result_compression_skipped") == 1


class _MockPolicy:
    def __init__(self) -> None:
        self.tool_levels = {"balanced": "balanced", "aggressive": "balanced", "full": "balanced"}
