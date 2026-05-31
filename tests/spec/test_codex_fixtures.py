"""Fixture-only Codex CLI adapter probe tests."""

from __future__ import annotations

import json
from pathlib import Path

from tok.runtime.types import ProcessedRuntimeResponse

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "codex"


def test_codex_probe_parses_fixture_request() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "request.json").read_bytes())

    assert adapter.identify_runtime() == "codex-cli"
    assert adapter.transport_boundary() == "http-proxy"
    assert request.model == "test-codex-model"
    assert request.surface_runtime == "codex-cli"
    assert request.surface_adapter == "codex-cli-probe"
    assert request.request_has_tools is True


def test_codex_probe_builds_fixture_response_shape() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    response = adapter.build_outbound_response(
        ProcessedRuntimeResponse(
            content_blocks=[{"type": "text", "text": "Use small changes and do not revert user work."}],
            output_saved_tokens=0,
            behavior_signals={},
            mode="tok",
            family_mode="tok",
            updated_memory="",
        ),
        conversation_id="codex-fixture-1",
    )

    assert json.loads(response) == json.loads((FIXTURE_DIR / "response_expected.json").read_text())


def test_codex_probe_capabilities_are_fixture_only() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    assert "fixture-only" in adapter.supported_capabilities()
    assert "http-proxy" in adapter.supported_capabilities()
    assert "live-codex-cli" not in adapter.supported_capabilities()


def test_codex_probe_labels_fixture_evidence_forms() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "exactness_labels.json").read_bytes())

    forms = [message["tok_probe"]["evidence_form"] for message in request.messages]
    assert forms == ["exact", "summary", "skeleton", "reference"]
    assert request.request_policy == "legacy_tool_compatible"
    assert request.tool_compatible is True


def test_codex_probe_emits_fallback_state_for_unknown_tool() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "fallback_state.json").read_bytes())

    assert request.request_policy == "forced_baseline"
    assert request.tool_compatible is False
    fallback_messages = [message for message in request.messages if message.get("tok_probe", {}).get("fallback")]
    assert fallback_messages


def test_codex_probe_handles_unsupported_capabilities_as_passthrough() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "unsupported_capabilities.json").read_bytes())

    assert request.request_policy == "forced_baseline"
    assert request.tool_compatible is False
    fallback = [message for message in request.messages if message.get("tok_probe", {}).get("fallback")][0]
    assert fallback["tok_probe"]["unsupported_capabilities"] == ["live-codex-cli"]


def test_codex_probe_exposes_diagnostic_signals_in_request_metadata() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "diagnostic_signals.json").read_bytes())

    assert request.todo is not None
    signals = json.loads(request.todo)["diagnostic_signals"]
    assert signals == {
        "adapter_probe_fallback": 0,
        "adapter_probe_parsed": 1,
        "adapter_probe_unsupported_capability": 0,
    }


def test_codex_probe_exposes_response_behavior_signals() -> None:
    from tok.adapters.probes import CodexAdapter

    adapter = CodexAdapter()
    response = adapter.build_outbound_response(
        ProcessedRuntimeResponse(
            content_blocks=[{"type": "text", "text": "Signals are preserved."}],
            output_saved_tokens=0,
            behavior_signals={"adapter_probe_response": 1},
            mode="tok",
            family_mode="tok",
            updated_memory="",
        )
    )

    assert json.loads(response)["tok"]["behavior_signals"] == {"adapter_probe_response": 1}
