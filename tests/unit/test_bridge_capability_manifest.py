from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, is_dataclass
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from tok.cli import app
from tok.gateway._types import BridgeCapabilityManifest, build_capability_manifest
from tok.spec.trace import TRACE_VERSION

runner = CliRunner()


def test_manifest_is_frozen_dataclass() -> None:
    manifest = build_capability_manifest()

    assert is_dataclass(manifest)
    with pytest.raises(FrozenInstanceError):
        manifest.bridge_mode = "baseline"  # type: ignore[misc]


def test_build_capability_manifest_returns_valid_defaults() -> None:
    manifest = build_capability_manifest()

    assert manifest.trace_version == TRACE_VERSION
    assert manifest.supported_evidence_forms == ("exact", "summary", "skeleton", "reference")
    assert "fallback" in manifest.supported_actions
    assert manifest.supported_delta_algorithms == ("unified_diff",)
    assert manifest.max_conformance_level == "L2"


def test_manifest_roundtrip_through_asdict() -> None:
    payload = asdict(build_capability_manifest())

    assert set(payload) == set(BridgeCapabilityManifest.__dataclass_fields__)
    assert payload["fixture_pack_version"] == version("tok-protocol")


def test_manifest_bridge_mode_overridable() -> None:
    assert build_capability_manifest(bridge_mode="baseline").bridge_mode == "baseline"


def test_bridge_status_prints_capability_manifest(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "status": "ok",
                "bridge": "tok",
                "port": 9090,
                "mode": "natural-first",
                "baseline_only": False,
                "fallback_count": 0,
                "session_tokens_saved": 12,
                "session_savings_pct": 10.0,
                "session_quality": "clean",
                "last_degradation_reason": "",
                "capability": asdict(build_capability_manifest(bridge_mode="natural-first")),
            }

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

    result = runner.invoke(app, ["bridge", "status"])

    assert result.exit_code == 0
    assert "Bridge capability" in result.output
    assert TRACE_VERSION in result.output
    assert "exact, summary, skeleton, reference" in result.output
