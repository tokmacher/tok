"""Resolver-backed path cache for exact evidence content.

This is a local-only helper layered on top of ``tok.resolver.store.ContentStore``.
It stores bytes content-addressably and keeps a small path manifest so callers can
retrieve the latest cached bytes for a path, optionally guarded by mtime.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tok.resolver.store import ContentStore


@dataclass(frozen=True)
class ResolverCacheEntry:
    path: str
    digest: str
    mtime: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"path": self.path, "digest": self.digest}
        if self.mtime is not None:
            data["mtime"] = self.mtime
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResolverCacheEntry | None:
        path = data.get("path")
        digest = data.get("digest")
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        raw_mtime = data.get("mtime")
        mtime = float(raw_mtime) if isinstance(raw_mtime, int | float) else None
        return cls(path=path, digest=digest, mtime=mtime)


class ResolverCache:
    """Content-addressed resolver cache indexed by normalized path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._store = ContentStore(root)
        self._manifest_path = root / "path_manifest.json"

    def put(self, *, path: str, content: bytes, mtime: float | None = None) -> str:
        digest = self._store.put(content)
        manifest = self._load_manifest()
        manifest[_normalize_path(path)] = ResolverCacheEntry(
            path=_normalize_path(path),
            digest=digest,
            mtime=mtime,
        )
        self._write_manifest(manifest)
        return digest

    def get(self, *, path: str, mtime: float | None = None) -> bytes | None:
        entry = self._load_manifest().get(_normalize_path(path))
        if entry is None:
            return None
        if mtime is not None and entry.mtime is not None and float(mtime) != entry.mtime:
            return None
        return self._store.get(entry.digest)

    def invalidate(self, *, path: str) -> None:
        manifest = self._load_manifest()
        manifest.pop(_normalize_path(path), None)
        self._write_manifest(manifest)

    def digest_for_path(self, *, path: str, mtime: float | None = None) -> str | None:
        entry = self._load_manifest().get(_normalize_path(path))
        if entry is None:
            return None
        if mtime is not None and entry.mtime is not None and float(mtime) != entry.mtime:
            return None
        return entry.digest

    def _load_manifest(self) -> dict[str, ResolverCacheEntry]:
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        entries: dict[str, ResolverCacheEntry] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            entry = ResolverCacheEntry.from_dict(value)
            if entry is not None:
                entries[_normalize_path(key)] = entry
        return entries

    def _write_manifest(self, manifest: dict[str, ResolverCacheEntry]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            key: entry.to_dict()
            for key, entry in sorted(manifest.items(), key=lambda item: item[0])
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix="tok-resolver-cache-",
            suffix=".json",
            dir=str(self._manifest_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._manifest_path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def _normalize_path(path: str) -> str:
    normalized = (path or "").strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


__all__ = ["ResolverCache", "ResolverCacheEntry"]
