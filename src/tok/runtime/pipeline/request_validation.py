"""Validation utilities for Tok runtime requests."""

import copy
import os
from typing import Any


_ALLOWED_BLOCK_TYPES = frozenset({"text", "tool_use", "tool_result"})


def _normalize_message_content_to_blocks(
    content: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize message content to a list of Anthropic-style blocks.

    Returns (blocks, drops) where drops maps dropped block type names to
    the count of blocks of that type that were removed because they are
    not in the outbound Anthropic allowlist.
    """
    if isinstance(content, str):
        text = content.strip()
        return ([{"type": "text", "text": text}] if text else []), {}
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        drops: dict[str, int] = {}
        for b in content:
            if not isinstance(b, dict):
                continue
            b_type = b.get("type", "")
            if b_type == "text":
                text = str(b.get("text", "")).strip()
                if text:
                    blocks.append({"type": "text", "text": text})
            elif b_type in _ALLOWED_BLOCK_TYPES:
                blocks.append(copy.deepcopy(b))
            else:
                drops[b_type] = drops.get(b_type, 0) + 1
        return blocks, drops
    return [], {}


def _check_changed_content(
    canonical_message: dict[str, Any],
    original_content: Any,
    role: str,
) -> bool:
    """Return True when normalization changed the message content."""
    if role == "tool_result":
        return False
    canonical_blocks = canonical_message.get("content")
    if not isinstance(canonical_blocks, list):
        canonical_blocks = []
    normalized_blocks, _ = _normalize_message_content_to_blocks(
        original_content
    )
    return canonical_blocks != normalized_blocks


def _canonicalize_bridge_message(
    message: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Rewrite a single message into a canonical user or assistant role with blocks.

    Returns (canonical_message, drops) where drops maps dropped block type
    names to counts of blocks removed during normalization.
    """
    role = str(message.get("role", "")).strip()
    if role == "tool_result":
        block = {
            "type": "tool_result",
            "tool_use_id": message.get("tool_use_id", ""),
            "content": copy.deepcopy(message.get("content", "")),
        }
        if "is_error" in message:
            block["is_error"] = bool(message.get("is_error"))
        if "cache_control" in message:
            block["cache_control"] = copy.deepcopy(
                message.get("cache_control")
            )

        return {"role": "user", "content": [block]}, {}

    canonical_role = "assistant" if role == "assistant" else "user"
    blocks, drops = _normalize_message_content_to_blocks(
        message.get("content")
    )
    return {"role": canonical_role, "content": blocks}, drops


def _merge_adjacent_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge adjacent messages of the same role (specifically for user messages)."""
    if not messages:
        return [], {}

    merged: list[dict[str, Any]] = []
    signals: dict[str, int] = {}

    for msg in messages:
        if not merged:
            merged.append(msg)
            continue

        prev = merged[-1]
        # Only merge user messages; assistant messages must not be merged across tool boundaries
        if msg["role"] == "user" and prev["role"] == "user":
            prev["content"].extend(msg["content"])
            signals["tok_bridge_adjacent_user_merged"] = (
                signals.get("tok_bridge_adjacent_user_merged", 0) + 1
            )
        else:
            merged.append(msg)

    return merged, signals


def _process_bridged_message(
    raw_message: dict[str, Any],
    signals: dict[str, int],
    total_drops: dict[str, int],
) -> tuple[dict[str, Any] | None, bool]:
    """Process a single message for canonicalization."""
    if not isinstance(raw_message, dict):
        return None, False

    role = str(raw_message.get("role", "")).strip()
    changed = False

    if role == "tool_result":
        signals["tok_bridge_top_level_tool_result_rewritten"] = (
            signals.get("tok_bridge_top_level_tool_result_rewritten", 0) + 1
        )
        changed = True

    orig_content = raw_message.get("content")
    msg, msg_drops = _canonicalize_bridge_message(raw_message)

    for b_type, count in msg_drops.items():
        total_drops[b_type] = total_drops.get(b_type, 0) + count
        changed = True

    if not msg["content"]:
        return None, True

    if not changed:
        changed = _check_changed_content(msg, orig_content, role)

    return msg, changed


def canonicalize_anthropic_bridge_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """Canonicalize bridge messages to the Anthropic wire shape.

    Returns (canonical_messages, changed, signals).

    Drops unsupported block types (anything outside {text, tool_use,
    tool_result}) and emits ``tok_bridge_unsupported_block_dropped`` and
    ``tok_bridge_thinking_block_dropped`` signals when blocks are removed.

    This function removes unsupported block types (e.g., top-level tool_result),
    rewrites top-level tool_results (emitting ``tok_bridge_top_level_tool_result_rewritten``),
    merges adjacent messages of same role (``tok_bridge_adjacent_messages_merged``),
    drops thinking blocks (since the bridge currently strips them internally vs
    tool_result) and emits ``tok_bridge_unsupported_block_dropped`` and
    ``tok_bridge_thinking_block_dropped`` signals when blocks are removed.
    """
    if not isinstance(messages, list):
        return messages, False, {}

    canonical_path: list[dict[str, Any]] = []
    signals: dict[str, int] = {}
    total_drops: dict[str, int] = {}

    for raw_message in messages:
        msg, changed = _process_bridged_message(
            raw_message, signals, total_drops
        )
        if msg is not None:
            canonical_path.append(msg)

    merged_messages, merge_signals = _merge_adjacent_anthropic_messages(
        canonical_path
    )
    signals.update(merge_signals)

    if not changed and (
        len(merged_messages) != len(messages) or merge_signals
    ):
        changed = True

    if total_drops:
        total_drop_count = sum(total_drops.values())
        signals["tok_bridge_unsupported_block_dropped"] = total_drop_count
        thinking_count = total_drops.get("thinking", 0) + total_drops.get(
            "redacted_thinking", 0
        )
        if thinking_count:
            signals["tok_bridge_thinking_block_dropped"] = thinking_count

    if changed:
        signals["tok_bridge_canonicalized"] = 1

    return merged_messages, changed, signals


def canonicalize_anthropic_bridge_body(
    body: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, int]]:
    """Canonicalize a bridge request body for Anthropic before send."""
    if not isinstance(body, dict):
        return body, False, {}
    messages = body.get("messages")
    if not isinstance(messages, list):
        return copy.deepcopy(body), False, {}

    (
        canonical_messages,
        changed,
        signals,
    ) = canonicalize_anthropic_bridge_messages(messages)
    if not changed:
        return body, False, {}

    new_body = copy.deepcopy(body)
    new_body["messages"] = canonical_messages
    return new_body, True, signals


def _process_assistant_tool_ids(
    content: Any,
    seen_tool_use_ids: set[str],
) -> set[str]:
    """Extract tool_use IDs from an assistant message."""
    assistant_tool_use_ids: set[str] = set()
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_use_id = str(block.get("id", "")).strip()
            if tool_use_id:
                assistant_tool_use_ids.add(tool_use_id)
                seen_tool_use_ids.add(tool_use_id)
    return assistant_tool_use_ids


def _process_user_tool_results(
    content: list[dict[str, Any]],
    seen_tool_use_ids: set[str],
    pending_tool_use_ids: set[str],
    risks: dict[str, int],
) -> None:
    """Check user text/tool_result ordering risks."""
    saw_text_block = False
    message_has_order_violation = False
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("type", "")).strip()
        if block_type == "text":
            if str(block.get("text", "")).strip():
                saw_text_block = True
            continue

        if block_type != "tool_result":
            continue

        if saw_text_block and not message_has_order_violation:
            risks["user_tool_result_after_text"] = (
                risks.get("user_tool_result_after_text", 0) + 1
            )
            message_has_order_violation = True

        tool_use_id = str(block.get("tool_use_id", "")).strip()
        if not tool_use_id:
            continue
        if tool_use_id not in seen_tool_use_ids:
            risks["tool_result_unknown_tool_use_id"] = (
                risks.get("tool_result_unknown_tool_use_id", 0) + 1
            )
        if tool_use_id not in pending_tool_use_ids:
            risks["tool_result_not_immediately_after_assistant_tool_use"] = (
                risks.get(
                    "tool_result_not_immediately_after_assistant_tool_use", 0
                )
                + 1
            )


def _collect_bridge_tool_result_shape_risks(
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    """Return Anthropic-specific tool_result ordering/pairing risks."""
    if not isinstance(messages, list):
        return {}

    risks: dict[str, int] = {}
    seen_tool_use_ids: set[str] = set()
    pending_tool_use_ids: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            pending_tool_use_ids = set()
            continue

        role = str(message.get("role", "")).strip()
        content = message.get("content")

        if role == "assistant":
            pending_tool_use_ids = _process_assistant_tool_ids(
                content, seen_tool_use_ids
            )
            continue

        if role != "user":
            pending_tool_use_ids = set()
            continue

        if isinstance(content, str) or not isinstance(content, list):
            pending_tool_use_ids = set()
            continue

        _process_user_tool_results(
            content, seen_tool_use_ids, pending_tool_use_ids, risks
        )
        pending_tool_use_ids = set()

    return risks


def _summarize_message_blocks(
    content: Any,
    summary: dict[str, Any],
) -> list[str]:
    """Summarize blocks within a message."""
    blocks_summary = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                blocks_summary.append("non_dict")
                continue
            b_type = b.get("type", "unknown")
            blocks_summary.append(str(b_type))
            if b_type == "tool_use":
                summary["tool_use_blocks"] += 1
            elif b_type == "tool_result":
                summary["tool_result_blocks"] += 1
            elif b_type not in _ALLOWED_BLOCK_TYPES:
                unsupported = summary["unsupported_blocks"]
                unsupported[b_type] = unsupported.get(b_type, 0) + 1
    elif isinstance(content, str):
        blocks_summary.append("str")
    else:
        blocks_summary.append("empty" if content is None else "unknown")
    return blocks_summary


def summarize_message_structure(
    messages: list[dict[str, Any]],
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
    for msg in messages:
        if not isinstance(msg, dict):
            role_seq.append(f"<{type(msg).__name__}>")
            continue

        role = str(msg.get("role", "none"))
        if role == "user":
            summary["user_msgs"] += 1
        elif role == "assistant":
            summary["assistant_msgs"] += 1

        content = msg.get("content")
        blocks_summary = _summarize_message_blocks(content, summary)

        role_seq.append(f"{role}[{','.join(blocks_summary)}]")

    summary["sequence"] = role_seq
    summary["field_shape_risks"] = _collect_bridge_tool_result_shape_risks(
        messages
    )
    return summary


def _validate_tool_use_block(
    block: dict[str, Any], role: str, failures: list[str]
) -> None:
    if role == "user":
        failures.append("user_contains_tool_use")
    if (
        not str(block.get("id", "")).strip()
        or not str(block.get("name", "")).strip()
        or not isinstance(block.get("input", {}), dict)
    ):
        failures.append("invalid_tool_use_block")


def _validate_tool_result_block(
    block: dict[str, Any], role: str, failures: list[str]
) -> None:
    if role == "assistant":
        failures.append("assistant_contains_tool_result")
    if not str(block.get("tool_use_id", "")).strip() or not isinstance(
        block.get("content", ""), str | list
    ):
        failures.append("invalid_tool_result_block")


def _validate_block(
    block: Any,
    role: str,
    msg_index: int,
    block_index: int,
    failures: list[str],
) -> None:
    """Validate a single block within a message."""
    if not isinstance(block, dict):
        failures.append(f"message_{msg_index}_block_{block_index}_not_dict")
        return

    block_type = str(block.get("type", "")).strip()
    if not block_type:
        failures.append(
            f"message_{msg_index}_block_{block_index}_missing_type"
        )
        return

    if block_type == "text":
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            failures.append("empty_message_content")
        return

    if block_type == "tool_use":
        _validate_tool_use_block(block, role, failures)
        return

    if block_type == "tool_result":
        _validate_tool_result_block(block, role, failures)
        return

    failures.append("unsupported_block_type")


def _validate_message(
    message: Any, msg_index: int, failures: list[str]
) -> None:
    """Validate a single message in the bridge body."""
    if not isinstance(message, dict):
        failures.append(f"message_{msg_index}_not_dict")
        return

    role = str(message.get("role", "")).strip()
    if role not in {"user", "assistant"}:
        failures.append("invalid_top_level_role")
        return

    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            failures.append("empty_message_content")
        return

    if not isinstance(content, list):
        failures.append("empty_message_content")
        return
    if not content:
        failures.append("empty_content_blocks")
        return

    for block_index, block in enumerate(content):
        _validate_block(block, role, msg_index, block_index, failures)


def validate_anthropic_bridge_body(body: dict[str, Any]) -> list[str]:
    """Strictly validate the Anthropic bridge wire shape after canonicalization."""
    failures: list[str] = []
    if not isinstance(body, dict):
        return ["body_not_dict"]

    if not str(body.get("model", "")).strip():
        failures.append("missing_model")

    messages = body.get("messages")
    if not isinstance(messages, list):
        return ["messages_not_list"]
    if not messages:
        return ["empty_messages"]

    for msg_index, message in enumerate(messages):
        _validate_message(message, msg_index, failures)

    shape_risks = _collect_bridge_tool_result_shape_risks(messages)
    if shape_risks.get("user_tool_result_after_text"):
        failures.append("user_tool_result_after_text")
    if shape_risks.get("tool_result_unknown_tool_use_id"):
        failures.append("tool_result_unknown_tool_use_id")
    if shape_risks.get("tool_result_not_immediately_after_assistant_tool_use"):
        failures.append("tool_result_not_immediately_after_assistant_tool_use")

    system = body.get("system")
    if system is not None and not isinstance(system, str | list):
        failures.append("invalid_system_type")
    return list(set(failures))  # Unique stable codes


def _is_valid_content_block(block: object) -> bool:
    """Helper for generic runtime validation."""
    if not isinstance(block, dict):
        return False
    block_type = str(block.get("type", "")).strip()
    if not block_type:
        return False
    if block_type == "text":
        return isinstance(block.get("text", ""), str)
    if block_type == "tool_use":
        return isinstance(block.get("name", ""), str) and isinstance(
            block.get("input", {}), dict
        )
    if block_type == "tool_result":
        return isinstance(block.get("tool_use_id", ""), str)
    return True


def _validate_message_basic(msg: Any, failures: list[str]) -> bool:
    """Validate a single message basic structure. Returns True if failed."""
    if not isinstance(msg, dict):
        failures.append("message_not_dict")
        return True
    if str(msg.get("role", "")).strip() not in {
        "user",
        "assistant",
        "tool_result",
        "system",
    }:
        failures.append("invalid_message_role")
        return True
    content = msg.get("content", "")
    if isinstance(content, list):
        if not all(_is_valid_content_block(block) for block in content):
            failures.append("invalid_content_block")
            return True
    elif not isinstance(content, str):
        failures.append("invalid_message_content")
        return True
    return False


def validate_anthropic_request_body(body: dict[str, Any]) -> list[str]:
    """Validate the structure of an Anthropic API request body.

    Returns a list of failure reason strings, or an empty list if valid.
    """
    failures: list[str] = []
    if not isinstance(body, dict):
        return ["body_not_dict"]
    if not str(body.get("model", "")).strip():
        failures.append("missing_model")
    messages = body.get("messages")
    if not isinstance(messages, list):
        failures.append("messages_not_list")
    else:
        if not messages:
            failures.append("empty_messages")
        for msg in messages:
            if _validate_message_basic(msg, failures):
                break
    system = body.get("system")
    if system is not None and not isinstance(system, str | list):
        failures.append("invalid_system_type")
    if isinstance(system, list):
        if not all(_is_valid_content_block(block) for block in system):
            failures.append("invalid_system_block")
    return failures


def detect_prompt_bloat(
    system_prompt: str | list[dict[str, Any]] | None, user_prompt: str = ""
) -> bool:
    """Identify when system prompts are unusually large or contain leaked user content.

    Returns True if the system prompt exceeds the TOK_PROMPT_BLOAT_THRESHOLD (default 2000)
    or if it appears to contain a substantial portion of the current user prompt.
    """
    if system_prompt is None:
        return False

    # Threshold for automatic optimization (chars)
    BLOAT_THRESHOLD = int(os.getenv("TOK_PROMPT_BLOAT_THRESHOLD", "2000"))

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = " ".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        system_text = str(system_prompt)

    if len(system_text) > BLOAT_THRESHOLD:
        return True

    # Check if user prompt content is leaking into system context (e.g. flattening)
    if user_prompt and len(user_prompt) > 200:
        # Check if a substantial part of the user prompt is in the system prompt
        snippet = user_prompt[:100].strip()
        if snippet and snippet in system_text:
            return True

    return False


def should_optimize_prompts(
    system_prompt: str | list[dict[str, Any]] | None,
    session_metrics: dict[str, int],
) -> bool:
    """Check if optimization is recommended based on size thresholds or size metrics."""
    # Threshold for intervention (chars)
    SIZE_LIMIT = int(os.getenv("TOK_PROMPT_OPTIMIZE_LIMIT", "2500"))

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = " ".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    elif system_prompt:
        system_text = str(system_prompt)

    if len(system_text) > SIZE_LIMIT:
        return True

    # Check for high growth rate signal if provided in metrics
    if session_metrics.get("tok_prompt_growth_high"):
        return True

    return detect_prompt_bloat(system_prompt)


__all__ = [
    "validate_anthropic_request_body",
    "canonicalize_anthropic_bridge_messages",
    "canonicalize_anthropic_bridge_body",
    "summarize_message_structure",
    "validate_anthropic_bridge_body",
    "detect_prompt_bloat",
    "should_optimize_prompts",
]
