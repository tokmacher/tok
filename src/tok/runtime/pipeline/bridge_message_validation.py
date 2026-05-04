"""Bridge message diagnostics shared by validation and gateway preflight."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DEFAULT_ALLOWED_BLOCK_TYPES = frozenset({"text", "tool_use", "tool_result", "thinking", "redacted_thinking"})
DEFAULT_PROVIDER_SENSITIVE_LARGE_TOOL_BATCH_THRESHOLD = 16


def collect_provider_sensitivity_risks(
    messages: list[dict[str, Any]],
    *,
    large_tool_batch_threshold: int = DEFAULT_PROVIDER_SENSITIVE_LARGE_TOOL_BATCH_THRESHOLD,
) -> dict[str, int]:
    """Return provider-sensitive mixed assistant tool/text batch risks."""
    if not isinstance(messages, list):
        return {}

    risks: dict[str, int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        tool_positions = [
            block_index
            for block_index, block in enumerate(content)
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_positions:
            continue
        first_tool = tool_positions[0]
        tool_use_count = len(tool_positions)
        has_text_between_or_after_tool_uses = any(
            isinstance(block, dict) and block.get("type") == "text" and block_index > first_tool
            for block_index, block in enumerate(content)
        )
        if tool_use_count >= large_tool_batch_threshold:
            risks["assistant_large_tool_use_batch"] = risks.get("assistant_large_tool_use_batch", 0) + 1
        if has_text_between_or_after_tool_uses:
            risks["assistant_tool_use_text_interleaving"] = risks.get("assistant_tool_use_text_interleaving", 0) + 1
        if tool_use_count >= large_tool_batch_threshold and has_text_between_or_after_tool_uses:
            next_message = messages[index + 1] if index + 1 < len(messages) else None
            next_content = next_message.get("content") if isinstance(next_message, dict) else None
            next_tool_result_count = 0
            if isinstance(next_content, list):
                next_tool_result_count = sum(
                    1 for block in next_content if isinstance(block, dict) and block.get("type") == "tool_result"
                )
            risks["assistant_large_tool_use_text_interleaving"] = (
                risks.get("assistant_large_tool_use_text_interleaving", 0) + 1
            )
            if next_tool_result_count:
                risks["provider_sensitive_large_tool_use_text_interleaving"] = (
                    risks.get("provider_sensitive_large_tool_use_text_interleaving", 0) + 1
                )
            else:
                risks["provider_sensitive_large_tool_use_batch_unterminated"] = (
                    risks.get("provider_sensitive_large_tool_use_batch_unterminated", 0) + 1
                )

    return risks


def _summarize_message_blocks(
    content: str | list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    allowed_block_types: set[str] | frozenset[str],
) -> list[str]:
    blocks_summary = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                blocks_summary.append("non_dict")
                continue
            block_type = block.get("type", "unknown")
            blocks_summary.append(str(block_type))
            if block_type == "tool_use":
                summary["tool_use_blocks"] += 1
            elif block_type == "tool_result":
                summary["tool_result_blocks"] += 1
            elif block_type not in allowed_block_types:
                unsupported = summary["unsupported_blocks"]
                unsupported[block_type] = unsupported.get(block_type, 0) + 1
    elif isinstance(content, str):
        blocks_summary.append("str")
    else:
        blocks_summary.append("empty" if content is None else "unknown")
    return blocks_summary


def summarize_message_structure(
    messages: list[dict[str, Any]],
    *,
    allowed_block_types: set[str] | frozenset[str] = DEFAULT_ALLOWED_BLOCK_TYPES,
    shape_risk_collector: Callable[[list[dict[str, Any]]], dict[str, int]] | None = None,
    provider_risk_collector: Callable[[list[dict[str, Any]]], dict[str, int]] = collect_provider_sensitivity_risks,
) -> str | dict[str, Any]:
    """Return a compact structural summary safe for bridge diagnostics."""
    if not isinstance(messages, list):
        return f"invalid_messages_type:{type(messages).__name__}"

    summary: dict[str, Any] = {
        "count": len(messages),
        "sequence": [],
        "user_msgs": 0,
        "assistant_msgs": 0,
        "tool_use_blocks": 0,
        "tool_result_blocks": 0,
        "unsupported_blocks": {},
        "field_shape_risks": {},
    }

    role_seq: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            role_seq.append(f"<{type(message).__name__}>")
            continue

        role = str(message.get("role", "none"))
        if role == "user":
            summary["user_msgs"] += 1
        elif role == "assistant":
            summary["assistant_msgs"] += 1

        content = message.get("content")
        typed_content: str | list[dict[str, Any]] = content if isinstance(content, str | list) else []
        blocks_summary = _summarize_message_blocks(
            typed_content,
            summary,
            allowed_block_types=allowed_block_types,
        )
        role_seq.append(f"{role}[{','.join(blocks_summary)}]")

    summary["sequence"] = role_seq
    summary["field_shape_risks"] = shape_risk_collector(messages) if shape_risk_collector else {}
    summary["provider_sensitivity_risks"] = provider_risk_collector(messages)
    return summary


def summarize_bridge_pairing(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return assistant->next-user tool pairing snapshots for bridge diagnostics."""
    if not isinstance(messages, list):
        return []
    timeline: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        tool_use_ids = [
            str(block.get("id", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_use_ids:
            continue
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        next_role = str(next_message.get("role", "")).strip() if isinstance(next_message, dict) else "<none>"
        next_content = next_message.get("content") if isinstance(next_message, dict) else None
        next_tool_result_ids: list[str] = []
        if isinstance(next_content, list):
            next_tool_result_ids = [
                str(block.get("tool_use_id", "")).strip()
                for block in next_content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
        timeline.append(
            {
                "assistant_index": index,
                "next_role": next_role,
                "tool_use_ids": tool_use_ids,
                "next_tool_result_ids": next_tool_result_ids,
            }
        )
    return timeline
