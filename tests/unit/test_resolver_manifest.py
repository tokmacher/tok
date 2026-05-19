from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    from tok.resolver.manifest import ResolverManifest

    manifest = ResolverManifest.default()
    path = tmp_path / "manifest.tok"
    manifest.save(path)

    loaded = ResolverManifest.load(path)
    assert loaded.manifest_version == "tok-resolver/v0.1"
    assert loaded.supported_hash_algorithms == ("sha256",)
    assert loaded.routing_scope == "local_only"
    assert loaded.privacy.allow_remote_resolution is False


def test_manifest_rejects_remote_resolution(tmp_path: Path) -> None:
    from tok.resolver.manifest import ResolverManifest

    path = tmp_path / "manifest.tok"
    bad = ResolverManifest.default().to_dict()
    bad["privacy"]["allow_remote_resolution"] = True
    path.write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="allow_remote_resolution"):
        ResolverManifest.load(path)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("routing_scope", "remote_allowed", "routing_scope"),
        ("max_conformance_level", "L3b", "max_conformance_level"),
    ],
)
def test_manifest_rejects_non_local_scope_or_level(field: str, bad_value: str, message: str) -> None:
    from tok.resolver.manifest import ResolverManifest

    bad = ResolverManifest.default().to_dict()
    bad[field] = bad_value

    with pytest.raises(ValueError, match=message):
        ResolverManifest.from_dict(bad)


def test_manifest_load_rejects_corrupted_json(tmp_path: Path) -> None:
    from tok.resolver.manifest import ResolverManifest

    path = tmp_path / "manifest.tok"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="resolver_manifest_invalid"):
        ResolverManifest.load(path)


def test_manifest_rejects_empty_resolver_id(tmp_path: Path) -> None:
    from tok.resolver.manifest import ResolverManifest

    bad = ResolverManifest.default().to_dict()
    bad["resolver_id"] = ""
    with pytest.raises(ValueError, match="resolver_id"):
        ResolverManifest.from_dict(bad)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("max_total_bytes", None, "max_total_bytes"),
        ("max_total_bytes", 0, "max_total_bytes"),
        ("max_total_bytes", True, "max_total_bytes"),
        ("object_ttl_seconds", None, "object_ttl_seconds"),
        ("object_ttl_seconds", 0, "object_ttl_seconds"),
        ("object_ttl_seconds", True, "object_ttl_seconds"),
        ("eviction_policy", "fifo", "eviction_policy"),
    ],
)
def test_manifest_rejects_invalid_storage_policy(field: str, bad_value: object, message: str) -> None:
    from tok.resolver.manifest import ResolverManifest

    bad = ResolverManifest.default().to_dict()
    bad["storage_policy"][field] = bad_value

    with pytest.raises(ValueError, match=message):
        ResolverManifest.from_dict(bad)


def test_manifest_rejects_missing_storage_policy() -> None:
    from tok.resolver.manifest import ResolverManifest

    bad = ResolverManifest.default().to_dict()
    del bad["storage_policy"]

    with pytest.raises(ValueError, match="storage_policy"):
        ResolverManifest.from_dict(bad)
