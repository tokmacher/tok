from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_translate_classify import TranslateClassifyResult, prepare_translate_classify
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


def _make_body(**overrides) -> dict:
    defaults = dict(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hello"}],
    )
    defaults.update(overrides)
    return defaults


class TestTranslateClassifyResultDefaults:
    def test_step3_result_has_correct_defaults(self) -> None:
        r = TranslateClassifyResult()
        assert r.body == {}
        assert r.plan_finalization_turn is False
        assert r.behavior_signals == {}
        assert r.id_to_context == {}
        assert r.normalized_tool_events == []
        assert r.broad_audit_batch is False
        assert r.edit_reacquisition_signals == {}
        assert r.should_skip_history is False
        assert r.skip_reason == ""
        assert r.history_skip_reason == ""
        assert r.injected_state_payload == ""
        assert r.exact_search_evidence_keys_in_request == set()
        assert r.suppress_reacquisition_once is False
        assert r.stream_recovery_history_floor_active is False
        assert r.runtime_hints == []
        assert r.translated_messages == []
        assert r.context_dependency.depends_on_context is False

    def test_step3_result_all_fields_present(self) -> None:
        expected = {
            "body",
            "plan_finalization_turn",
            "behavior_signals",
            "id_to_context",
            "normalized_tool_events",
            "broad_audit_batch",
            "edit_reacquisition_signals",
            "should_skip_history",
            "skip_reason",
            "history_skip_reason",
            "injected_state_payload",
            "exact_search_evidence_keys_in_request",
            "suppress_reacquisition_once",
            "stream_recovery_history_floor_active",
            "runtime_hints",
            "translated_messages",
            "context_dependency",
        }
        actual = {f.name for f in fields(TranslateClassifyResult)}
        assert actual == expected


class TestPrepareTranslateClassify:
    def test_basic_request_translates_messages(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.translated_messages == [{"role": "user", "content": "hello"}]
        assert result.body["messages"] == result.translated_messages

    def test_behavior_signals_initialized(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert isinstance(result.behavior_signals, dict)

    def test_plan_finalization_turn_false_for_non_bridge(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request(adapter_kind="unknown")
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.plan_finalization_turn is False

    def test_plan_finalization_turn_true_for_bridge_with_keyword(self) -> None:
        session = RuntimeSession()
        body = _make_body(
            messages=[
                {"role": "assistant", "content": "here is a plan"},
                {"role": "user", "content": "finalize the plan"},
            ]
        )
        req = _make_request(
            adapter_kind="claude-bridge",
            messages=[
                {"role": "assistant", "content": "here is a plan"},
                {"role": "user", "content": "finalize the plan"},
            ],
        )
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=True)
        assert result.plan_finalization_turn is True

    def test_plan_handoff_uses_context_dependency_not_phrase_finalization(self) -> None:
        session = RuntimeSession()
        messages = [
            {"role": "assistant", "content": "Plan: inspect, patch, test.\n- read\n- edit\n- verify"},
            {"role": "user", "content": "proceed with your plan"},
        ]
        body = _make_body(messages=messages)
        req = _make_request(adapter_kind="claude-bridge", messages=messages)

        result = prepare_translate_classify(req, session, body, is_bridge_adapter=True)

        assert result.plan_finalization_turn is False
        assert result.context_dependency.depends_on_context is True
        assert result.context_dependency.kind == "plan_handoff"
        assert result.context_dependency.protected_suffix_start == 0
        assert result.behavior_signals.get("context_dependency_kind_plan_handoff") == 1
        assert result.behavior_signals.get("plan_finalization_turn") == 1

    def test_skip_history_defaults_false(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.should_skip_history is False
        assert result.skip_reason == ""
        assert result.history_skip_reason == ""

    def test_empty_tool_events_for_no_tool_messages(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.normalized_tool_events == []

    def test_broad_audit_batch_false_for_non_bridge(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request(adapter_kind="unknown")
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.broad_audit_batch is False

    def test_injected_state_payload_defaults_empty(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.injected_state_payload == ""

    def test_project_markers_proxy_in_signals(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert "_project_markers_proxy" in result.behavior_signals

    def test_stream_recovery_budget_decremented(self) -> None:
        session = RuntimeSession()
        session._stream_recovery_history_floor_budget = 2
        body = _make_body()
        req = _make_request()
        result = prepare_translate_classify(req, session, body, is_bridge_adapter=False)
        assert result.stream_recovery_history_floor_active is True
        assert session._stream_recovery_history_floor_budget == 1
