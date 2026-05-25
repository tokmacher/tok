"""Fixture-only Codex CLI adapter probe tests."""

from __future__ import annotations

import json
from pathlib import Path

from tok.runtime.types import ProcessedRuntimeResponse

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "codex"


def test_codex_probe_parses_fixture_request() -> None:
    from tok.adapters.codex_probe import CodexAdapter

    adapter = CodexAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "request.json").read_bytes())

    assert adapter.identify_runtime() == "codex-cli"
    assert adapter.transport_boundary() == "stdio"
    assert request.model == "test-codex-model"
    assert request.surface_runtime == "codex-cli"
    assert request.surface_adapter == "codex-cli-probe"
    assert request.request_has_tools is True


def test_codex_probe_builds_fixture_response_shape() -> None:
    from tok.adapters.codex_probe import CodexAdapter

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
    from tok.adapters.codex_probe import CodexAdapter

    adapter = CodexAdapter()
    assert "fixture-only" in adapter.supported_capabilities()
    assert "live-codex-cli" not in adapter.supported_capabilities()
