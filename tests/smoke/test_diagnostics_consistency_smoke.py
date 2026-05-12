from __future__ import annotations

from tok.runtime._diagnostics import DiagnosticsSnapshot


def test_diagnostics_consistency_smoke() -> None:
    payload = DiagnosticsSnapshot(
        port=9090,
        api_base="https://example.invalid",
        mode="tool-compatible",
        request_policy="natural_first",
        calls=3,
        fallback_count=1,
        fail_open_count=2,
        session_tokens_saved=100,
    ).to_health_response()
    snap = DiagnosticsSnapshot.from_health_response(payload)
    assert snap.to_health_response() == payload
    assert snap.calls == 3
    assert snap.fallback_count == 1
    assert snap.fail_open_count == 2
    assert snap.session_tokens_saved == 100
