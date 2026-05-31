"""Internal crypto helpers for protocol packages."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def _digest_file(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _digest_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()
