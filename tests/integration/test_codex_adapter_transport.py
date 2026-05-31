from __future__ import annotations

import json

import pytest

from tok.gateway._adapter_proxy import handle_adapter_proxy_request
from tok.runtime.core import RuntimeSession


def test_codex_adapter_transport_routes_codex_request_through_runtime(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    captured_provider_body: dict[str, object] = {}

    def _provider(body: dict[str, object]) -> dict[str, object]:
        captured_provider_body.update(body)
        return {"choices": [{"message": {"content": "File=src/tok/runtime/core.py\nVerification=ok"}}]}

    result = handle_adapter_proxy_request(
        json.dumps(
            {
                "conversation_id": "codex-live-1",
                "model": "test-model",
                "input": [
                    {"role": "user", "content": "Inspect runtime core"},
                    {"role": "tool_result", "tool": "read_file", "path": "src/tok/runtime/core.py", "content": "exact"},
                ],
            }
        ).encode(),
        provider_forwarder=_provider,
        session=RuntimeSession(),
    )

    payload = json.loads(result.body)
    assert result.status_code == 200
    assert result.fallback is False
    assert result.diagnostics["execution_path"] == "adapter-proxy"
    assert result.diagnostics["surface_runtime"] == "codex-cli"
    assert captured_provider_body["model"] == "test-model"
    assert payload["conversation_id"] == "codex-live-1"
    assert payload["tok"]["runtime"] == "codex-cli"
    assert payload["tok"]["behavior_signals"]["adapter_proxy_request"] == 1


def test_codex_adapter_transport_fail_opens_on_malformed_request(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    raw = b"{not-json"

    result = handle_adapter_proxy_request(raw, session=RuntimeSession())

    assert result.status_code == 200
    assert result.body == raw
    assert result.fallback is True
    assert result.diagnostics["fallback_reason"] == "adapter_parse_failed"
    assert result.diagnostics["execution_path"] == "raw_passthrough"


def test_codex_adapter_transport_reports_fallback_signal_for_unsupported_capability(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")

    result = handle_adapter_proxy_request(
        json.dumps(
            {
                "conversation_id": "codex-unsupported",
                "model": "test-model",
                "capabilities": ["live-codex-cli"],
                "input": [{"role": "user", "content": "hello"}],
            }
        ).encode(),
        session=RuntimeSession(),
    )

    assert result.fallback is False
    assert result.diagnostics["request_policy"] == "forced_baseline"
    signals = result.diagnostics["behavior_signals"]
    assert signals["adapter_probe_fallback"] == 1
    assert signals["adapter_probe_unsupported_capability"] == 1


def test_codex_adapter_transport_rejected_without_unstable_gate(monkeypatch) -> None:
    monkeypatch.delenv("TOK_UNSTABLE_ADAPTERS", raising=False)

    with pytest.raises(ValueError, match="TOK_UNSTABLE_ADAPTERS=1"):
        handle_adapter_proxy_request(b'{"model":"test-model","input":[]}', session=RuntimeSession())
