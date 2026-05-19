from __future__ import annotations

import os
from pathlib import Path

from tok.resolver.store import ContentStore, format_resolver_uri
from tok.spec.trace import audit_block, canonical_payload_digest


def _block(*, uri: str, digest: str, size: int) -> dict:
    block = {
        "envelope": {
            "trace_version": "tok-trace/v0.1-draft",
            "block_id": "b1",
            "session_id": "s1",
            "turn": 0,
            "step": 0,
            "direction": "request",
            "payload_digest": "draft-uncomputed",
        },
        "observation": {
            "class": "file",
            "key": "k",
            "action": "store",
            "result": "ok",
        },
        "content": {
            "exact": True,
            "hash": digest,
            "size_bytes": size,
            "resolver_uri": uri,
        },
        "audit": {
            "resolver_state": "available_local",
            "expectation": "accept_exact",
            "reason": "",
        },
    }
    block["envelope"]["payload_digest"] = canonical_payload_digest(block)
    return block


def test_audit_resolves_tok_resolver_uri(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path / "resolver"))
    store = ContentStore(Path(os.environ["TOK_RESOLVER_ROOT"]))
    digest = store.put(b"hello")
    uri = format_resolver_uri(digest)

    result = audit_block(_block(uri=uri, digest=digest, size=5))
    assert result.status == "pass"


def test_audit_fails_when_resolver_uri_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path / "resolver"))
    digest = "sha256:" + "0" * 64
    uri = format_resolver_uri(digest)

    result = audit_block(_block(uri=uri, digest=digest, size=5))
    assert result.status == "fail"
    assert "available_local_unresolved_content_uri" in result.errors
