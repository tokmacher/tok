"""Phase 0.2.5: live adapter configuration gating tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


def test_claude_adapter_allowed_by_default(monkeypatch) -> None:
    from tok.adapters.adapter_config import resolve_adapter_config

    monkeypatch.delenv("TOK_UNSTABLE_ADAPTERS", raising=False)
    config = resolve_adapter_config("claude")
    assert config.name == "claude"
    assert config.bridge_mode == "claude"


def test_opencode_rejected_without_unstable(monkeypatch) -> None:
    from tok.adapters.adapter_config import resolve_adapter_config

    monkeypatch.delenv("TOK_UNSTABLE_ADAPTERS", raising=False)
    try:
        resolve_adapter_config("opencode")
    except ValueError as exc:
        assert "TOK_UNSTABLE_ADAPTERS=1" in str(exc)
    else:
        raise AssertionError("opencode should require TOK_UNSTABLE_ADAPTERS=1")


def test_codex_rejected_without_unstable(monkeypatch) -> None:
    from tok.adapters.adapter_config import resolve_adapter_config

    monkeypatch.delenv("TOK_UNSTABLE_ADAPTERS", raising=False)
    try:
        resolve_adapter_config("codex-cli")
    except ValueError as exc:
        assert "TOK_UNSTABLE_ADAPTERS=1" in str(exc)
    else:
        raise AssertionError("codex-cli should require TOK_UNSTABLE_ADAPTERS=1")


def test_unstable_adapters_allowed_with_env(monkeypatch) -> None:
    from tok.adapters.adapter_config import resolve_adapter_config

    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    assert resolve_adapter_config("opencode").name == "opencode"
    assert resolve_adapter_config("codex-cli").name == "codex-cli"


def test_bridge_start_adapter_flag_rejects_unstable(monkeypatch) -> None:
    monkeypatch.delenv("TOK_UNSTABLE_ADAPTERS", raising=False)
    result = runner.invoke(app, ["bridge", "start", "--adapter", "opencode", "--foreground"])
    assert result.exit_code == 1
    assert "TOK_UNSTABLE_ADAPTERS=1" in result.output


def test_bridge_start_non_claude_adapter_rejected_even_with_unstable(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    result = runner.invoke(app, ["bridge", "start", "--adapter", "opencode", "--foreground"])
    assert result.exit_code == 1
    assert "Claude Code bridge path" in result.output


def test_opencode_fixture_roundtrip_with_unstable(monkeypatch) -> None:
    from tok.adapters.adapter_config import get_adapter_probe
    from tok.runtime.types import ProcessedRuntimeResponse

    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    adapter = get_adapter_probe("opencode")
    request = adapter.parse_inbound_request(b'{"model":"m","messages":[]}')
    assert request.surface_runtime == "opencode"
    response = adapter.build_outbound_response(
        ProcessedRuntimeResponse(
            content_blocks=[{"type": "text", "text": "ok"}],
            output_saved_tokens=0,
            behavior_signals={},
            mode="tok",
            family_mode="tok",
            updated_memory="",
        )
    )
    assert json.loads(response)["tok"]["fixture_only"] is True


def test_adapter_parse_fail_open_returns_none(monkeypatch) -> None:
    from tok.adapters.adapter_config import get_adapter_probe, parse_or_fail_open

    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    adapter = get_adapter_probe("codex-cli")
    assert parse_or_fail_open(adapter, b"{not-json") is None
