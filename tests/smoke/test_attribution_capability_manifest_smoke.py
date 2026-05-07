from __future__ import annotations

from fastapi.testclient import TestClient

from tok.gateway import BridgeSession, create_app
from tok.spec.trace import TRACE_VERSION


def test_health_endpoint_includes_capability_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    app = create_app(BridgeSession(memory_dir=tmp_path / ".tok", api_base="https://example.invalid"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    capability = response.json()["capability"]
    assert capability["trace_version"] == TRACE_VERSION
    assert capability["bridge_mode"]
    assert capability["max_conformance_level"] == "L2"
    assert "exact" in capability["supported_evidence_forms"]
