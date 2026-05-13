from __future__ import annotations

from pathlib import Path

import pytest


def test_object_path_rejects_malformed_digest(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        store._object_path("nope")


def test_put_roundtrips_bytes(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    digest = store.put(b"hello")
    assert digest.startswith("sha256:")
    assert store.has(digest) is True
    assert store.get(digest) == b"hello"


def test_get_returns_none_for_missing_digest(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    assert store.get("sha256:" + "0" * 64) is None


def test_get_raises_for_malformed_digest(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        store.get("sha256:xyz")


def test_has_returns_false_for_malformed_digest(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    assert store.has("nope") is False


def test_get_detects_tampered_bytes(tmp_path: Path) -> None:
    from tok.resolver.store import ContentStore

    store = ContentStore(tmp_path)
    digest = store.put(b"hello")
    tamper_path = store._object_path(digest)
    tamper_path.write_bytes(b"goodbye")
    with pytest.raises(ValueError, match="Digest mismatch"):
        store.get(digest)


def test_resolver_uri_roundtrip() -> None:
    from tok.resolver.store import format_resolver_uri, parse_resolver_uri

    digest = "sha256:" + "0" * 64
    uri = format_resolver_uri(digest)
    assert uri == f"tok-resolver://{digest}"
    assert parse_resolver_uri(uri) == digest


def test_resolver_uri_rejects_non_digest() -> None:
    from tok.resolver.store import parse_resolver_uri

    with pytest.raises(ValueError, match="Unsupported resolver URI"):
        parse_resolver_uri("file:///tmp/nope")
