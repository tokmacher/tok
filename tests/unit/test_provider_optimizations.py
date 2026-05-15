from __future__ import annotations

from tok.provider_optimizations import apply_provider_optimizations


def test_apply_provider_optimizations_is_noop_for_unknown_provider() -> None:
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    rewritten, saved = apply_provider_optimizations(adapter_kind="unknown", body=dict(body))
    assert rewritten == body
    assert saved == 0
