"""Provider-neutral classification of assistant/user content blocks.

Tok's request/response-shaping paths constantly need to ask "what kind of content
block is this?" -- is it a tool call, a tool result, plain text? Historically each
call site hardcoded Anthropic ``type`` string literals (``block.get("type") ==
"tool_use"`` and friends), scattering provider-shape knowledge across the codebase.

This module is the single classification surface for those questions. Routing the
shaping paths through it keeps block-shape knowledge in one place, so a future
provider whose wire format labels blocks differently can be supported by remapping
here (or behind an adapter) rather than editing dozens of call sites. It is the
non-opaque companion to :mod:`tok.provider_opaque_blocks`.

All helpers are deliberately defensive: they never raise on malformed shapes and
return ``False`` / ``None`` for anything that is not a well-formed block dict, so
they are safe to call at any validation or preflight boundary.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TEXT_BLOCK_TYPE",
    "TOOL_RESULT_BLOCK_TYPE",
    "TOOL_USE_BLOCK_TYPE",
    "block_type",
    "is_text_block",
    "is_tool_result_block",
    "is_tool_use_block",
]

# Canonical block-``type`` labels for the currently supported provider wire format
# (Anthropic Messages). Centralized here so a provider adapter can remap them in a
# single place instead of editing every comparison site.
TOOL_USE_BLOCK_TYPE = "tool_use"
TOOL_RESULT_BLOCK_TYPE = "tool_result"
TEXT_BLOCK_TYPE = "text"


def block_type(block: Any) -> str | None:
    """Return the block's ``type`` string, or ``None`` for malformed shapes."""
    if not isinstance(block, dict):
        return None
    value = block.get("type")
    return value if isinstance(value, str) else None


def is_tool_use_block(block: Any) -> bool:
    """Return ``True`` when *block* is a tool-call block."""
    return block_type(block) == TOOL_USE_BLOCK_TYPE


def is_tool_result_block(block: Any) -> bool:
    """Return ``True`` when *block* is a tool-result block."""
    return block_type(block) == TOOL_RESULT_BLOCK_TYPE


def is_text_block(block: Any) -> bool:
    """Return ``True`` when *block* is a plain-text block."""
    return block_type(block) == TEXT_BLOCK_TYPE
