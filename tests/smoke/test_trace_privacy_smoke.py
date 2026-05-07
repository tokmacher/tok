from __future__ import annotations

import json

from tok.spec.live_trace import emit_live_trace


class _BridgeMemory:
    turn = 1


class _RuntimeSession:
    bridge_memory = _BridgeMemory()


class _Session:
    _active_session_key = "test-session"
    _live_trace_instance_id = "test-trace-instance"

    def __init__(self, memory_dir) -> None:  # type: ignore[no-untyped-def]
        self.memory_dir = memory_dir
        self.runtime_session = _RuntimeSession()


def test_trace_privacy_smoke(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.delenv("TOK_TRACE_FILE", raising=False)
    monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)

    session = _Session(tmp_path)
    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="pass_through",
        result="ok",
        expectation="accept_pass_through",
        reason="smoke",
        direction="request",
        metadata={"compressed": False},
    )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert trace_files
    text = trace_files[0].read_text()
    for pattern in ("api.anthropic.com", "api.openai.com", "openrouter.ai"):
        assert pattern not in text
    for line in text.splitlines():
        if not line.strip():
            continue
        block = json.loads(line)
        session_id = block.get("envelope", {}).get("session_id", "")
        assert not str(session_id).startswith("http")
