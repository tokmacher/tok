"""Utilities for selecting safe history slices during request preparation."""

from __future__ import annotations

from typing import Any


def _message_has_tool_result(message: dict[str, Any]) -> bool:
    if message.get("role") == "tool_result":
        return bool(str(message.get("tool_use_id", "")).strip())
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _message_has_user_prompt(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip():
            return True
    return False


def _stream_recovery_winnowing_floor_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve minimal causal context after stream recovery retries."""
    if not messages:
        return []

    latest_user_prompt_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user" and _message_has_user_prompt(msg):
            latest_user_prompt_idx = idx
            break

    assistant_idx = -1
    assistant_tool_ids: set[str] = set()
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        tool_ids = {
            str(block.get("id", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use" and str(block.get("id", "")).strip()
        }
        if tool_ids:
            assistant_idx = idx
            assistant_tool_ids = tool_ids
            break

    paired_user_idx = -1
    if assistant_idx >= 0:
        for idx in range(assistant_idx + 1, len(messages)):
            msg = messages[idx]
            if msg.get("role") not in {"user", "tool_result"} or not _message_has_tool_result(msg):
                continue
            tool_result_ids: set[str]
            if msg.get("role") == "tool_result":
                tool_result_id = str(msg.get("tool_use_id", "")).strip()
                tool_result_ids = {tool_result_id} if tool_result_id else set()
            else:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                tool_result_ids = {
                    str(block.get("tool_use_id", "")).strip()
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and str(block.get("tool_use_id", "")).strip()
                }
            if assistant_tool_ids & tool_result_ids:
                paired_user_idx = idx
                break

    keep_indexes = sorted(idx for idx in {assistant_idx, paired_user_idx, latest_user_prompt_idx} if idx >= 0)
    return [messages[idx] for idx in keep_indexes]


def _messages_contain_tool_material(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") == "tool_result":
            return True
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"tool_use", "tool_result"}:
                return True
    return False


def _assistant_tool_use_ids(message: dict[str, Any]) -> set[str]:
    if str(message.get("role", "")).strip() != "assistant":
        return set()
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    return {
        str(block.get("id", "")).strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use" and str(block.get("id", "")).strip()
    }


def _message_tool_result_ids(message: dict[str, Any]) -> set[str]:
    if str(message.get("role", "")).strip() == "tool_result":
        tool_use_id = str(message.get("tool_use_id", "")).strip()
        return {tool_use_id} if tool_use_id else set()
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    return {
        str(block.get("tool_use_id", "")).strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result" and str(block.get("tool_use_id", "")).strip()
    }


def _bridge_recent_suffix_has_safe_pairing(messages: list[dict[str, Any]]) -> bool:
    """
    Validate immediate assistant tool_use -> next-user tool_result pairing.

    This is a lightweight preflight guard to skip known-bad cut candidates
    before canonicalization/validation.

    When the suffix starts with a tool_result-only user message (parallel
    agent mode), verify that every tool_result references a tool_use ID
    present in a preceding assistant message within the suffix.
    """
    if not messages:
        return False
    first = messages[0]
    if not isinstance(first, dict) or str(first.get("role", "")).strip() != "user":
        return False
    if _message_has_tool_result(first):
        return _tool_result_only_suffix_has_safe_pairing(messages)

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return False
        tool_ids = _assistant_tool_use_ids(message)
        if not tool_ids:
            continue
        if index + 1 >= len(messages):
            return False
        next_message = messages[index + 1]
        if not isinstance(next_message, dict) or str(next_message.get("role", "")).strip() != "user":
            return False
        if not _message_has_tool_result(next_message):
            return False
        next_content = next_message.get("content")
        if isinstance(next_content, list):
            if any(not (isinstance(block, dict) and block.get("type") == "tool_result") for block in next_content):
                return False
        result_ids = _message_tool_result_ids(next_message)
        if not tool_ids.issubset(result_ids):
            return False
    return True


def _tool_result_only_suffix_has_safe_pairing(messages: list[dict[str, Any]]) -> bool:
    """
    Validate pairing for suffixes that start with tool_result-only user messages.

    Allows dangling tool_results at the start (results whose tool_uses were
    in the cut-off portion of history). Only requires that every tool_use
    within the suffix has a corresponding tool_result within the suffix.
    """
    all_tool_use_ids: set[str] = set()
    all_tool_result_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        if role == "assistant":
            all_tool_use_ids |= _assistant_tool_use_ids(message)
        elif role == "user":
            all_tool_result_ids |= _message_tool_result_ids(message)
    if not all_tool_use_ids or not all_tool_result_ids:
        return False
    if not all_tool_use_ids.issubset(all_tool_result_ids):
        return False
    return True


def _bridge_preflight_safe_recent_suffix(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """
    Advance candidate history cut to the next user-starting safe boundary.

    When all user messages contain tool_results (e.g., parallel agent mode),
    fall back to the earliest tool-result-only user message whose suffix has
    safe pairing, rather than returning None and forcing a full fallback.
    """
    tool_result_only_fallback: list[dict[str, Any]] | None = None
    for start_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "user":
            continue
        if _message_has_tool_result(message):
            if tool_result_only_fallback is None:
                candidate = messages[start_index:]
                if _bridge_recent_suffix_has_safe_pairing(candidate):
                    tool_result_only_fallback = candidate
            continue
        candidate = messages[start_index:]
        if _bridge_recent_suffix_has_safe_pairing(candidate):
            return candidate
    return tool_result_only_fallback
