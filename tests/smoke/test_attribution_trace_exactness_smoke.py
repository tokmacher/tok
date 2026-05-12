from __future__ import annotations

import json
from pathlib import Path

from tok.spec.live_trace import emit_live_trace


class _BridgeMemory:
    turn = 1


class _RuntimeSession:
    bridge_memory = _BridgeMemory()


class _Session:
    _active_session_key = "attribution-smoke"
    _live_trace_instance_id = "attribution-smoke-trace"

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.runtime_session = _RuntimeSession()


def test_attribution_trace_exactness_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="skeleton_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="attribution smoke",
        metadata={"compressed": True, "reason": "skeleton"},
    )

    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    blocks = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]

    assert any(block["observation"]["action"] != "pass_through" for block in blocks)
    assert all(
        not block["content"]["exact"]
        for block in blocks
        if block["observation"]["action"] in {"skeleton_reference", "summary_reference"}
    )
