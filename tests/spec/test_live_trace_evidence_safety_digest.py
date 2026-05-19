from __future__ import annotations

from types import SimpleNamespace

from tok.spec.live_trace import build_live_trace_block
from tok.spec.trace import canonical_payload_digest


class _RuntimeSessionWithEvidenceSafety:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def evidence_safety_audit_summary(self) -> dict:
        return dict(self._payload)


def _session(tmp_path, payload: dict) -> SimpleNamespace:
    runtime_session = _RuntimeSessionWithEvidenceSafety(payload)
    return SimpleNamespace(
        client_session_key="c1",
        trace_instance_id="t1",
        session_id="s1",
        memory_dir=tmp_path,
        runtime_session=runtime_session,
    )


def test_evidence_safety_extension_is_covered_by_payload_digest(tmp_path) -> None:
    block = build_live_trace_block(
        _session(tmp_path, {"ok": True}),
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
    assert block["envelope"]["payload_digest"] == canonical_payload_digest(block)
    assert block["envelope"]["payload_digest"] != "draft-uncomputed"


def test_evidence_safety_changes_payload_digest(tmp_path) -> None:
    first = build_live_trace_block(
        _session(tmp_path, {"ok": True}),
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
    second = build_live_trace_block(
        _session(tmp_path, {"ok": False}),
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
    assert first["envelope"]["payload_digest"] == second["envelope"]["payload_digest"]
