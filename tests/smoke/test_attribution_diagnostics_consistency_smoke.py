from __future__ import annotations

from tok.runtime._diagnostics import DiagnosticsSnapshot


def test_attribution_diagnostics_consistency_smoke() -> None:
    snapshot = DiagnosticsSnapshot(
        calls=5,
        fallback_count=1,
        session_tokens_saved=42,
        fail_open_count=1,
        api_base="https://example.invalid",
        mode="natural-first",
    )

    health = snapshot.to_health_response()
    roundtrip = DiagnosticsSnapshot.from_health_response(health)

    assert roundtrip.calls == 5
    assert roundtrip.fallback_count == 1
    assert roundtrip.session_tokens_saved == 42
    assert roundtrip.fail_open_count == 1
    assert roundtrip.to_health_response() == health
