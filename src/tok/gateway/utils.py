"""Gateway HTTP utilities extracted from gateway for modularity."""

from __future__ import annotations

import json

import httpx

_DROP_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
}


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    }


def _payloads_materially_differ(
    content: bytes, original_content: bytes | None
) -> bool:
    if original_content is None or original_content == content:
        return False
    try:
        current = json.loads(content)
        original = json.loads(original_content)
    except Exception:
        return True
    if isinstance(current, dict) and isinstance(original, dict):
        current = dict(current)
        original = dict(original)
        if current.get("system", None) == "" and "system" not in original:
            current.pop("system", None)
        if original.get("system", None) == "" and "system" not in current:
            original.pop("system", None)
    return bool(current != original)
