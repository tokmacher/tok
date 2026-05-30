"""Central classification of provider-owned *opaque* content blocks.

Some assistant content blocks are produced and cryptographically signed by the
upstream provider (e.g. Anthropic ``thinking`` and ``redacted_thinking`` blocks).
These blocks are **opaque** to Tok: their bytes, fields, and position within the
assistant message carry provider-internal meaning that Tok cannot interpret,
summarize, or reconstruct. The provider rejects any request whose latest
assistant message contains opaque blocks that were modified, reordered, dropped,
deduplicated, or re-encoded lossily:

    API Error: 400 messages.N.content.M: thinking or redacted_thinking blocks in
    the latest assistant message cannot be modified. These blocks must remain as
    they were in the original response.

Because Tok forwards request history upstream as a bridge, it MUST treat these
blocks as immutable pass-through data. No request-transforming path may compress,
summarize, reorder, merge, dedupe, strip, restore-from-summary, or re-serialize
them in a lossy way.

This module is the single source of truth for *which* blocks are opaque so every
request-shaping path (canonicalization, block-order normalization, compression,
fail-open retry/fallback reconstruction, etc.) agrees and stays provider-aware
rather than hardcoding ``"thinking"`` checks at each call site.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "OPAQUE_PROVIDER_BLOCK_TYPES",
    "content_has_opaque_provider_blocks",
    "is_opaque_provider_block",
]

# Provider-owned block ``type`` values that must never be modified or dropped
# when forwarding request history upstream. This is intentionally a set so that
# future providers which mark blocks opaque can extend it here, in one place,
# rather than special-casing individual transformation sites.
OPAQUE_PROVIDER_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def is_opaque_provider_block(block: Any) -> bool:
    """Return ``True`` when *block* is a provider-owned opaque content block.

    Opaque blocks are detected by:

    1. A known opaque ``type`` (``thinking`` / ``redacted_thinking``), or
    2. The presence of a non-empty ``signature`` field. A provider signature is
       a strong, provider-agnostic signal that the block is signed reasoning the
       provider owns; mutating or dropping it invalidates the signature.

    The check is deliberately conservative and never raises on malformed shapes
    so it can be called safely at any request-validation / preflight boundary.
    """
    if not isinstance(block, dict):
        return False
    block_type = block.get("type")
    if isinstance(block_type, str) and block_type in OPAQUE_PROVIDER_BLOCK_TYPES:
        return True
    # A signed block is provider-signed reasoning even if a future provider uses
    # a different ``type`` label; preserve it byte-for-byte regardless.
    return bool(block.get("signature"))


def content_has_opaque_provider_blocks(content: Any) -> bool:
    """Return ``True`` when *content* is a block list holding any opaque block."""
    if not isinstance(content, list):
        return False
    return any(is_opaque_provider_block(block) for block in content)
