"""Cache control helpers for gateway."""

from __future__ import annotations


def _body_has_cache_control(value: object) -> bool:
    """Return True if cache_control appears anywhere in a body subtree."""
    if isinstance(value, dict):
        if "cache_control" in value:
            return True
        return any(_body_has_cache_control(item) for item in value.values())
    if isinstance(value, list):
        return any(_body_has_cache_control(item) for item in value)
    return False


def _count_cache_control_entries(value: object) -> int:
    """Count cache_control occurrences anywhere in a body subtree."""
    if isinstance(value, dict):
        count = 1 if "cache_control" in value else 0
        return count + sum(
            _count_cache_control_entries(item) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_count_cache_control_entries(item) for item in value)
    return 0


def _cache_control_counts_for_messages(messages: object) -> dict[str, int]:
    """Count cache_control placement by Anthropic message block type."""
    counts = {
        "message_text_blocks": 0,
        "message_tool_result_blocks": 0,
        "message_tool_use_blocks": 0,
    }
    if not isinstance(messages, list):
        return counts

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if "cache_control" not in block:
                continue
            block_type = str(block.get("type", "")).strip()
            if block_type == "text":
                counts["message_text_blocks"] += 1
            elif block_type == "tool_result":
                counts["message_tool_result_blocks"] += 1
            elif block_type == "tool_use":
                counts["message_tool_use_blocks"] += 1

    return counts


def _cache_control_counts_for_tools(tools: object) -> int:
    """Count cache_control markers on the top-level tools array."""
    if not isinstance(tools, list):
        return 0
    return sum(
        1
        for tool in tools
        if isinstance(tool, dict) and "cache_control" in tool
    )
