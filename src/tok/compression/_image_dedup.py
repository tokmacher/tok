"""Image block deduplication for tool_result content lists.

Replaces already-seen base64 image blocks with a compact text stub,
saving the token cost of re-encoding large images in long conversations.
URL-sourced images are never fingerprinted or stubbed.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _image_fingerprint(block: dict[str, Any]) -> str:
    """Return a 16-hex-char SHA-256 prefix of the image's base64 data."""
    source = block.get("source", {})
    data = source.get("data", "")
    digest = hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()
    return digest[:16]


def strip_duplicate_images(
    content: list[dict[str, Any]],
    seen_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Replace previously-seen base64 image blocks with a text stub.

    Mutates *seen_fingerprints* in place so callers accumulate state across
    multiple tool_result blocks within the same request.

    Returns the (possibly modified) content list and the total chars saved.
    URL-sourced images are never touched.
    """
    if not content:
        return content, 0

    result: list[dict[str, Any]] = []
    saved = 0

    for block in content:
        if block.get("type") != "image":
            result.append(block)
            continue

        source = block.get("source", {})
        if source.get("type") != "base64":
            # URL or other source — pass through unchanged
            result.append(block)
            continue

        fp = _image_fingerprint(block)
        data = source.get("data", "")
        original_chars = len(data) if isinstance(data, str) else len(str(data))

        if fp in seen_fingerprints:
            stub_text = f"[image: {fp}, already delivered at prior turn]"
            result.append({"type": "text", "text": stub_text})
            saved += max(0, original_chars - len(stub_text))
        else:
            seen_fingerprints.add(fp)
            result.append(block)

    return result, saved
