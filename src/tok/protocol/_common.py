"""Internal common helpers for protocol packages."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _prefixed_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex
