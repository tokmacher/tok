"""Shared private utilities for the adapters package."""

from __future__ import annotations

from typing import Any


def _system_to_messages(
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert system prompt to list of message dicts."""
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    return [
        {"role": "system", "content": str(block["text"])}
        for block in system
        if isinstance(block, dict) and block.get("text")
    ]


def _render_text(content_blocks: list[dict[str, Any]]) -> str:
    """Extract and join text from content blocks."""
    return "\n".join(
        t for block in content_blocks if block.get("type") == "text" and (t := str(block.get("text", "")).strip())
    )
