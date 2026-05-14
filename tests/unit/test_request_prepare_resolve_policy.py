from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_resolve_policy import Step4Result, run_step_4
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


class TestStep4ResultDefaults:
    def test_step4_result_has_correct_defaults(self) -> None:
        r = Step4Result()
        assert r.effective_tool_compatible is False
        assert r.request_policy_reasons == []
        assert r.request_policy_escalated is False
        assert r.behavior_signals == {}
        assert r.mode is None
        assert r.policy is None
        assert r.current_pressure == 0
        assert r.hot_hint_metrics == {}
        assert r.saved_tokens == 0
        assert r.type_breakdown == {}
        assert r.should_skip_history is False
        assert r.skip_reason == ""
        assert r.history_skip_reason == ""
        assert r.request_policy == ""

    def test_step4_result_all_fields_present(self) -> None:
        expected = {
            "effective_tool_compatible",
            "request_policy_reasons",
            "request_policy_escalated",
            "behavior_signals",
            "mode",
            "policy",
            "current_pressure",
            "hot_hint_metrics",
            "saved_tokens",
            "type_breakdown",
            "should_skip_history",
            "skip_reason",
            "history_skip_reason",
            "request_policy",
        }
        actual = {f.name for f in fields(Step4Result)}
        assert actual == expected


class TestStep4ResolvePolicy:
    def test_legacy_tool_compatible_effective_true(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            request_policy="legacy_tool_compatible",
            tool_compatible=True,
            messages=[
                {"role": "assistant", "content": "use tools"},
                {"role": "user", "content": "follow up"},
            ],
        )
        result = run_step_4(
            request=req,
            session=session,
            translated_messages=req.messages,
            normalized_tool_events=[],
            behavior_signals={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            plan_finalization_turn=False,
        )
        assert result.effective_tool_compatible is True

    def test_forced_baseline_effective_false(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            request_policy="forced_baseline",
            tool_compatible=True,
            messages=[{"role": "user", "content": "hello"}],
        )
        result = run_step_4(
            request=req,
            session=session,
            translated_messages=req.messages,
            normalized_tool_events=[],
            behavior_signals={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            plan_finalization_turn=False,
        )
        assert result.effective_tool_compatible is False

    def test_natural_first_without_messages_stays_natural_first(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            request_policy="natural_first",
            tool_compatible=True,
            messages=[{"role": "user", "content": "hello"}],
        )
        result = run_step_4(
            request=req,
            session=session,
            translated_messages=[],
            normalized_tool_events=[],
            behavior_signals={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            plan_finalization_turn=False,
        )
        assert result.request_policy == "natural_first"

    def test_behavior_signals_populated_after_resolution(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            request_policy="legacy_tool_compatible",
            tool_compatible=True,
            messages=[
                {"role": "assistant", "content": "use tools"},
                {"role": "user", "content": "follow up"},
            ],
        )
        behavior_signals: dict[str, int] = {}
        result = run_step_4(
            request=req,
            session=session,
            translated_messages=req.messages,
            normalized_tool_events=[],
            behavior_signals=behavior_signals,
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            plan_finalization_turn=False,
        )
        assert len(result.behavior_signals) > 0
