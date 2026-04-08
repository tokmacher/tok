"""Fingerprint helpers for gateway diagnostics."""

from __future__ import annotations

from typing import Any


def _get_header_value(headers: dict[str, str], name: str) -> str:
    """Read a header from a plain dict with case-insensitive lookup."""
    name_lower = name.lower()
    for key, value in headers.items():
        if key.lower() == name_lower:
            return str(value)
    return ""


def _request_uses_prompt_caching(headers: dict[str, str], body: dict[str, Any]) -> bool:
    """Return True when the inbound request uses Anthropic prompt caching."""
    beta_header = _get_header_value(headers, "anthropic-beta")
    if "prompt-caching" in beta_header:
        return True
    return bool(isinstance(body, dict) and _body_has_cache_control(body))


def _system_fingerprint(system: object) -> dict[str, int | str]:
    """Summarize system shape without logging raw content."""
    if system is None:
        return {
            "type": "missing",
            "block_count": 0,
            "text_length": 0,
            "cache_control_blocks": 0,
        }
    if isinstance(system, str):
        return {
            "type": "str",
            "block_count": 1 if system else 0,
            "text_length": len(system),
            "cache_control_blocks": 0,
        }
    if isinstance(system, list):
        text_length = 0
        cache_control_blocks = 0
        for block in system:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_length += len(str(block.get("text", "")))
            if "cache_control" in block:
                cache_control_blocks += 1
        return {
            "type": "list",
            "block_count": len(system),
            "text_length": text_length,
            "cache_control_blocks": cache_control_blocks,
        }
    return {
        "type": type(system).__name__,
        "block_count": 0,
        "text_length": len(str(system)),
        "cache_control_blocks": 0,
    }


def _cache_control_counts_for_messages(messages: object) -> dict[str, int]:
    """Count cache_control placement by Anthropic message block type."""
    from ._cache_control import (
        _cache_control_counts_for_messages as _counts_helper,
    )

    return _counts_helper(messages)


def _cache_control_counts_for_tools(tools: object) -> int:
    """Count cache_control markers on the top-level tools array."""
    from ._cache_control import (
        _cache_control_counts_for_tools as _tools_helper,
    )

    return _tools_helper(tools)


def _request_body_fingerprint(headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    """Build a redacted request fingerprint for prompt-caching diagnostics."""
    system = _system_fingerprint(body.get("system"))
    cache_counts = _cache_control_counts_for_messages(body.get("messages"))
    cache_counts["system_blocks"] = int(system["cache_control_blocks"])
    cache_counts["tools"] = _cache_control_counts_for_tools(body.get("tools"))
    cache_counts["total"] = (
        _cache_control_counts_for_messages(body.get("messages"))["message_text_blocks"]
        + cache_counts["message_tool_result_blocks"]
        + cache_counts["message_tool_use_blocks"]
        + cache_counts["system_blocks"]
        + cache_counts["tools"]
    )

    known = (
        cache_counts["system_blocks"]
        + cache_counts["message_text_blocks"]
        + cache_counts["message_tool_result_blocks"]
        + cache_counts["message_tool_use_blocks"]
        + cache_counts["tools"]
    )
    cache_counts["other"] = max(0, cache_counts["total"] - known)

    beta_header = _get_header_value(headers, "anthropic-beta")
    return {
        "anthropic_beta": beta_header or "<unset>",
        "prompt_caching": ("prompt-caching" in beta_header or cache_counts["total"] > 0),
        "system": system,
        "cache_control": cache_counts,
    }


def _system_value_for_compare(body: dict[str, Any]) -> object:
    """Treat missing system and empty-string system equivalently for compare."""
    has_system = "system" in body
    system = body.get("system")
    if system == "" and not has_system:
        return None
    return system


def _body_has_cache_control(value: object) -> bool:
    """Return True if cache_control appears anywhere in a body subtree."""
    from ._cache_control import _body_has_cache_control as _cache_check

    return _cache_check(value)
