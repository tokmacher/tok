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
