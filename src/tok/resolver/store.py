from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContentStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _object_path(self, digest: str) -> Path:
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"Invalid digest: {digest!r} (expected sha256:<64 lowercase hex>)")
        hex_digest = digest.split(":", 1)[1]
        prefix = hex_digest[:2]
        rest = hex_digest[2:]
        return self._root / "objects" / prefix / rest

    def has(self, digest: str) -> bool:
        if not _SHA256_RE.fullmatch(digest):
            return False
        return self._object_path(digest).is_file()

    def get(self, digest: str) -> bytes | None:
        path = self._object_path(digest)
        if not path.is_file():
            return None
        data = path.read_bytes()
        expected = digest.split(":", 1)[1]
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"Digest mismatch for {digest}: expected {expected}, got {actual}")
        return data

    def put(self, data: bytes) -> str:
        hex_digest = hashlib.sha256(data).hexdigest()
        digest = f"sha256:{hex_digest}"
        final_path = self._object_path(digest)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(prefix="tok-obj-", dir=str(final_path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, final_path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
        return digest


def parse_resolver_uri(uri: str) -> str:
    if not uri.startswith("tok-resolver://"):
        raise ValueError(f"Unsupported resolver URI: {uri!r}")
    digest = uri.removeprefix("tok-resolver://")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"Invalid resolver URI digest: {digest!r}")
    return digest


def format_resolver_uri(digest: str) -> str:
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"Invalid digest: {digest!r}")
    return f"tok-resolver://{digest}"
