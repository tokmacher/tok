"""Fixture-only OpenCode adapter probe tests."""

from __future__ import annotations

import json
from pathlib import Path

from tok.runtime.types import ProcessedRuntimeResponse

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "opencode"


def test_opencode_probe_parses_fixture_request() -> None:
    from tok.adapters.probes import OpenCodeAdapter

    adapter = OpenCodeAdapter()
    request = adapter.parse_inbound_request((FIXTURE_DIR / "request.json").read_bytes())

    assert adapter.identify_runtime() == "opencode"
    assert adapter.transport_boundary() == "stdio"
    assert request.model == "test-model"
    assert request.surface_runtime == "opencode"
    assert request.surface_adapter == "opencode-probe"
    assert request.request_has_tools is True


def test_opencode_probe_builds_fixture_response_shape() -> None:
    from tok.adapters.probes import OpenCodeAdapter

    adapter = OpenCodeAdapter()
    response = adapter.build_outbound_response(
        ProcessedRuntimeResponse(
            content_blocks=[{"type": "text", "text": "Tok stays bridge-first."}],
            output_saved_tokens=0,
            behavior_signals={},
            mode="tok",
            family_mode="tok",
            updated_memory="",
        ),
        response_id="opencode-fixture-1",
    )

    assert json.loads(response) == json.loads((FIXTURE_DIR / "response_expected.json").read_text())


def test_opencode_probe_capabilities_are_fixture_only() -> None:
    from tok.adapters.probes import OpenCodeAdapter

    adapter = OpenCodeAdapter()
    assert "fixture-only" in adapter.supported_capabilities()
    assert "live-opencode" not in adapter.supported_capabilities()
