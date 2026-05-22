"""System prompt static-section cache hint injection.

Detects the static prefix of a system prompt across turns and applies
Anthropic cache_control hints to it, making subsequent reads significantly
cheaper (cached tokens cost ~10x less than fresh input tokens).

Content is never removed from the system prompt — only cache hints are added.
"""

from __future__ import annotations

import hashlib
from typing import Any

_MIN_CACHEABLE_CHARS = 200
_FP_LENGTH = 16


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:_FP_LENGTH]


def paragraph_fingerprints(text: str) -> list[str]:
    """Split on blank lines and return a SHA-256 prefix per paragraph."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [_fingerprint(p) for p in paragraphs]


def find_static_prefix_length(
    current_fps: list[str],
    prior_fps: list[str],
) -> int:
    """Return the count of leading fingerprints that match between current and prior."""
    count = 0
    for c, p in zip(current_fps, prior_fps, strict=False):
        if c != p:
            break
        count += 1
    return count


def _split_text_at_paragraph(text: str, n_paragraphs: int) -> tuple[str, str]:
    """Split text at the nth paragraph boundary (0-indexed count of blank-line-separated blocks)."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    static = "\n\n".join(paragraphs[:n_paragraphs])
    dynamic = "\n\n".join(paragraphs[n_paragraphs:])
    return static, dynamic


def apply_system_cache_hint(
    system: str | list[dict[str, Any]],
    prior_fingerprints: list[str] | None,
) -> tuple[str | list[dict[str, Any]], list[str], int]:
    """Apply cache_control hints to the static prefix of the system prompt.

    Returns:
        (system_possibly_with_hints, current_fingerprints, estimated_static_chars)

    On first turn (prior_fingerprints is None): returns system unchanged, with
    fingerprints of current content and static_chars=0.

    On subsequent turns with a stable prefix of at least _MIN_CACHEABLE_CHARS:
    converts the static prefix block to a cached content block and leaves the
    dynamic suffix without cache_control.
    """
    # Resolve the canonical text for fingerprinting
    if isinstance(system, str):
        text_for_fp = system
    else:
        # List of content blocks: fingerprint the concatenated text content
        text_for_fp = "\n\n".join(block.get("text", "") for block in system if isinstance(block, dict))

    current_fps = paragraph_fingerprints(text_for_fp)

    if prior_fingerprints is None:
        return system, current_fps, 0

    static_count = find_static_prefix_length(current_fps, prior_fingerprints)
    if static_count == 0:
        return system, current_fps, 0

    # String case: split into static + dynamic blocks
    if isinstance(system, str):
        static_text, dynamic_text = _split_text_at_paragraph(system, static_count)
        if len(static_text) < _MIN_CACHEABLE_CHARS:
            return system, current_fps, 0

        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": static_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if dynamic_text:
            blocks.append({"type": "text", "text": dynamic_text})
        return blocks, current_fps, len(static_text)

    # List case: apply cache_control to the first block if it's large enough
    if not system:
        return system, current_fps, 0

    first = system[0]
    if not isinstance(first, dict) or first.get("type") != "text":
        return system, current_fps, 0

    first_text = first.get("text", "")
    if len(first_text) < _MIN_CACHEABLE_CHARS:
        return system, current_fps, 0

    if "cache_control" in first:
        # Already hinted — idempotent
        return system, current_fps, len(first_text)

    new_first = {**first, "cache_control": {"type": "ephemeral"}}
    return [new_first, *system[1:]], current_fps, len(first_text)
