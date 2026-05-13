"""Tests for warm-start injection on first request."""

from __future__ import annotations

import json

from tok.runtime.core import RuntimeRequest, RuntimeSession, UniversalTokRuntime


def test_warm_start_injected_on_first_request_when_prior_state_exists() -> None:
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    session.bridge_memory.turn = 10
    session.bridge_memory.bump_file_heat("src/a.py", weight=2.0)

    prepared = runtime.prepare_request(
        RuntimeRequest(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}]), session
    )

    payload = json.dumps(prepared.body)
    assert "tok warm-start - may be stale" in payload
