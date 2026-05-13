from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_resolver_returns_wrong_bytes_detected(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    digest = store.put(b"hello")
    object_path = store._object_path(digest)
    object_path.write_bytes(b"goodbye")

    with pytest.raises(ValueError, match="Digest mismatch"):
        store.get(digest)


def test_hash_exists_but_content_missing_returns_none(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    assert store.get("sha256:" + "0" * 64) is None


def test_resolver_uri_path_traversal_rejected() -> None:
    from tok.resolver.store import parse_resolver_uri

    with pytest.raises(ValueError):
        parse_resolver_uri("tok-resolver://sha256:../../etc/passwd" + "0" * 46)


def test_available_local_claimed_but_absent_fails_audit(tmp_path: Path, monkeypatch) -> None:
    from tok.resolver.store import format_resolver_uri
    from tok.spec.trace import audit_block, canonical_payload_digest

    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path / "resolver"))
    digest = "sha256:" + "0" * 64
    uri = format_resolver_uri(digest)
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
        "observation": {"class": "file", "key": "k", "action": "store", "result": "ok"},
        "content": {"exact": True, "hash": digest, "size_bytes": 5, "resolver_uri": uri},
        "audit": {"resolver_state": "available_local", "expectation": "accept_exact", "reason": "test"},
    }
    block["envelope"]["payload_digest"] = canonical_payload_digest(block)
    result = audit_block(block)
    assert result.status == "fail"
    assert "available_local_unresolved_content_uri" in result.errors


def test_adversarial_pack_cases_have_expected_outcomes() -> None:
    path = Path("docs/spec/fixtures/resolver_local_beta_adversarial.json")
    data = json.loads(path.read_text())
    pack = data["packs"][0]
    assert pack["status"] == "implemented-local"
    for case in pack["cases"]:
        assert case["expected_status"] in {"pass", "warn", "fail"}
        assert isinstance(case["expected_error"], str) and case["expected_error"]
