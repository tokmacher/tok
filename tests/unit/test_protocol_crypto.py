from __future__ import annotations

import json
from pathlib import Path

from tok.protocol._crypto import _digest_file, _digest_payload


def test_digest_file_returns_sha256_prefix(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    path.write_text("hello", encoding="utf-8")
    digest = _digest_file(path)
    assert digest.startswith("sha256:")
    assert len(digest) == 71  # "sha256:" + 64 hex chars


def test_digest_file_deterministic_same_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same content", encoding="utf-8")
    b.write_text("same content", encoding="utf-8")
    assert _digest_file(a) == _digest_file(b)


def test_digest_file_different_content_produces_different_digest(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("content A", encoding="utf-8")
    b.write_text("content B", encoding="utf-8")
    assert _digest_file(a) != _digest_file(b)


def test_digest_payload_returns_sha256_prefix() -> None:
    digest = _digest_payload({"key": "value"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_digest_payload_sort_key_independent() -> None:
    a = _digest_payload({"b": 1, "a": 2})
    b = _digest_payload({"a": 2, "b": 1})
    assert a == b


def test_digest_payload_uses_compact_serialization() -> None:
    payload = {"a": 1, "b": 2}
    digest = _digest_payload(payload)
    assert digest.startswith("sha256:")
    expected_encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    from hashlib import sha256

    expected_digest = "sha256:" + sha256(expected_encoded).hexdigest()
    assert digest == expected_digest
