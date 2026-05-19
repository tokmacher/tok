from __future__ import annotations

from types import SimpleNamespace

from tok.spec.live_trace import build_live_trace_block


def test_live_trace_never_emits_draft_uncomputed_digest(tmp_path) -> None:
    session = SimpleNamespace(
        client_session_key="c1",
        trace_instance_id="t1",
        session_id="s1",
        memory_dir=tmp_path,
        runtime_session=None,
    )
    block = build_live_trace_block(
        session,
        "test-event",
        trace_class="file",
        action="store",
        result="ok",
        expectation="accept_pass_through",
        reason="test",
        direction="request",
        metadata={"k": "v"},
        trace_file=tmp_path / "trace.jsonl",
    )
    assert block["envelope"]["payload_digest"] != "draft-uncomputed"
