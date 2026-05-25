from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_optimize_prompt import OptimizePromptResult, prepare_optimize_prompt
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


def _make_body(system: str | None = None) -> dict:
    body: dict = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if system is not None:
        body["system"] = system
    return body


class TestOptimizePromptResultDefaults:
    def test_optimize_prompt_result_defaults(self) -> None:
        r = OptimizePromptResult()
        assert r.body == {}
        assert r.compressed is False

    def test_optimize_prompt_result_all_fields(self) -> None:
        expected = {"body", "compressed"}
        actual = {f.name for f in fields(OptimizePromptResult)}
        assert actual == expected


class TestPrepareOptimizePrompt:
    def test_no_bloat_no_change(self) -> None:
        session = RuntimeSession()
        body = _make_body(system="short prompt")
        req = _make_request()
        result = prepare_optimize_prompt(req, session, body, "hello", False, False)
        assert result.compressed is False
        assert result.body["system"] == "short prompt"

    def test_no_system_no_change(self) -> None:
        session = RuntimeSession()
        body = _make_body()
        req = _make_request()
        result = prepare_optimize_prompt(req, session, body, "hello", False, False)
        assert result.compressed is False

    def test_bridge_adapter_with_bloat_records_signal(self) -> None:
        session = RuntimeSession()
        long_system = "x" * 3000
        body = _make_body(system=long_system)
        req = _make_request(adapter_kind="claude-bridge")
        prepare_optimize_prompt(req, session, body, "hello", True, False)
        assert session.pending_behavior_signals.get("tok_prompt_bloat_detected") == 1
        assert session.pending_behavior_signals.get("tok_prompt_optimization_skipped_bridge") == 1

    def test_non_bridge_with_bloat_can_optimize(self) -> None:
        session = RuntimeSession()
        long_system = "x" * 3000
        body = _make_body(system=long_system)
        req = _make_request(adapter_kind="unknown")
        prepare_optimize_prompt(req, session, body, "hello", False, False)
        assert session.pending_behavior_signals.get("tok_prompt_bloat_detected") == 1

    def test_compressed_preserved_when_no_bloat(self) -> None:
        session = RuntimeSession()
        body = _make_body(system="short")
        req = _make_request()
        result = prepare_optimize_prompt(req, session, body, "hello", False, True)
        assert result.compressed is True

    def test_compressed_set_when_optimization_applied(self) -> None:
        session = RuntimeSession()
        long_system = "System prompt. " * 500
        body = _make_body(system=long_system)
        req = _make_request(adapter_kind="unknown")
        result = prepare_optimize_prompt(req, session, body, "hello", False, False)
        if result.body.get("system") != long_system:
            assert result.compressed is True
