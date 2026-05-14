from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_inject_system import Step8Result, run_step_8
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


class TestStep8ResultDefaults:
    def test_step8_result_has_correct_defaults(self) -> None:
        r = Step8Result()
        assert r.body == {}
        assert r.injected_state_payload == ""
        assert r.runtime_hints == []
        assert r.behavior_signals == {}
        assert r.hot_hint_metrics == {}
        assert r.resend_signals == {}
        assert r.answer_ready is False
        assert r.has_answer_anchor is False
        assert r.session_memory == ""
        assert r.tok_state == ""

    def test_step8_result_all_fields_present(self) -> None:
        expected = {
            "body",
            "injected_state_payload",
            "runtime_hints",
            "behavior_signals",
            "hot_hint_metrics",
            "resend_signals",
            "answer_ready",
            "has_answer_anchor",
            "session_memory",
            "tok_state",
        }
        actual = {f.name for f in fields(Step8Result)}
        assert actual == expected


class TestStep8InjectSystem:
    def test_tool_compatible_false_calls_inject_system_additions(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.body == body

    def test_skip_reason_short_session_sets_signal(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="short_session",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.behavior_signals.get("short_session_system_additions_skipped") == 1

    def test_skip_reason_broad_audit_sets_signal(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="broad_audit",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.behavior_signals.get("broad_audit_system_additions_skipped") == 1

    def test_effective_tool_compatible_true_with_skip_returns_early(self) -> None:
        session = RuntimeSession()
        req = _make_request(tool_compatible=True)
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="short_session",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=True,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.body == body

    def test_empty_runtime_hints_passed_through(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.runtime_hints == []

    def test_empty_behavior_signals_passed_through(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.behavior_signals == {}

    def test_has_answer_anchor_default_false(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.has_answer_anchor is False

    def test_answer_ready_default_false(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.answer_ready is False

    def test_hot_hint_metrics_default_empty(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.hot_hint_metrics == {}

    def test_resend_signals_default_empty(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = run_step_8(
            runtime_self=None,
            request=req,
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        assert result.resend_signals == {}
