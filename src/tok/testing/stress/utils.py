"""Pure shared helpers for stress harness.

Functions here are used across multiple modules (executor, classification, reports, runner)
and should not introduce circular dependencies between modules.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from tok.runtime import RuntimeSession
from .models import (
    READ_ONLY_TOOL_NAMES,
)


def _system_to_messages(
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    messages: list[dict[str, Any]] = []
    for block in system:
        if isinstance(block, dict):
            text = str(block.get("text", "")).strip()
            if text:
                messages.append({"role": "system", "content": text})
    return messages


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = str(block.get("type", "")).strip()
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif block_type == "tool_use":
                name = str(block.get("name", "tool")).strip() or "tool"
                tool_input = json.dumps(block.get("input", {}), sort_keys=True)
                parts.append(f"Tool use ({name}): {tool_input}")
            elif block_type == "tool_result":
                tool_id = (
                    str(block.get("tool_use_id", "")).strip() or "unknown"
                )
                parts.append(
                    f"Tool result ({tool_id}): {_content_text(block.get('content', ''))}"
                )
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        if "content" in content:
            return _content_text(content.get("content"))
    return str(content)


def _normalize_chat_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = _content_text(message.get("content", ""))
        if not content.strip():
            continue
        if role == "tool_result":
            tool_id = str(message.get("tool_use_id", "")).strip() or "unknown"
            normalized.append(
                {
                    "role": "user",
                    "content": f"Tool result ({tool_id}): {content}",
                }
            )
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _render_visible_text(content_blocks: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(block.get("text", "")).strip()
        for block in content_blocks
        if block.get("type") == "text" and str(block.get("text", "")).strip()
    ).strip()


def _extract_labeled_fields(
    text: str, session: RuntimeSession | None = None
) -> dict[str, str]:
    fields: dict[str, str] = {}
    labels = ["file", "verification", "related"]
    for label in labels:
        pattern = rf"(?:\|>\s*)?{label}\s*[:=]\s*([^\s\n|]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            fields[label] = matches[-1].strip()

    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("|>"):
            cleaned = cleaned[2:].strip()
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        k = key.strip().lower()
        if k not in fields:
            fields[k] = value.strip()

    if session and session.bridge_memory:
        pointers = session.bridge_memory.pointers
        for key, val in fields.items():
            if str(val).startswith("*"):
                resolved = pointers.resolve(val)
                if resolved:
                    fields[key] = resolved
    return fields


def _strip_answer_labels(text: str) -> str:
    cleaned = str(text or "")
    for marker in ("File=", "Verification="):
        index = cleaned.find(marker)
        if index == -1:
            continue
        cleaned = cleaned[:index]
        break
    return cleaned.strip()


def _sanitize_tool_input(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = _strip_answer_labels(value)
        cleaned = re.sub(r"Tool use \([^)]+\):\s*", "", cleaned)
        return cleaned.strip()
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("Tool use ("):
                continue
            cleaned_dict[str(key)] = _sanitize_tool_input(item)
        return cleaned_dict
    if isinstance(value, list):
        return [_sanitize_tool_input(item) for item in value]
    return value


def _sanitize_tool_use_block(block: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(block)
    tool_input = block.get("input", {})
    sanitized["input"] = (
        _sanitize_tool_input(tool_input)
        if isinstance(tool_input, dict)
        else {}
    )
    return sanitized


def _normalize_extracted_path(path: str) -> str:
    cleaned = path.strip().replace(".tok/", "/tok/")
    if (
        cleaned.startswith("src/tok/")
        and "/tok/" in cleaned[len("src/tok/") :]
    ):
        nested = cleaned[cleaned.find("/tok/", len("src/tok/")) + 1 :]
        if nested.startswith("tok/"):
            cleaned = "src/" + nested
    match = re.search(r"(src/tok/[\w./-]+\.\w+)$", cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    data = {"role": message.get("role", "user")}
    if message.get("role") == "tool_result":
        data["tool_use_id"] = message.get("tool_use_id", "")
    data["content"] = _content_text(message.get("content", ""))
    return data


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fields_key(fields: dict[str, str]) -> str:
    return (
        f"{fields.get('file', '').strip().lower()}|"
        f"{fields.get('verification', '').strip().lower()}"
    )


def _is_supported_read_only_tool_name(name: str) -> bool:
    return str(name).strip().lower() in READ_ONLY_TOOL_NAMES


def _normalized_target_path(path: str) -> str:
    cleaned = str(path or "").strip().lower()
    return _normalize_extracted_path(cleaned) if cleaned else ""


def _tool_directly_reopens_expected_target(
    block: dict[str, Any], expected_fields: dict[str, str]
) -> bool:
    expected_file = _normalized_target_path(expected_fields.get("file", ""))
    expected_verification = (
        str(expected_fields.get("verification", "")).strip().lower()
    )
    if not expected_file and not expected_verification:
        return False
    name = str(block.get("name", "")).strip().lower()
    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}
    path_text = _normalized_target_path(
        tool_input.get("path") or tool_input.get("file_path") or ""
    )
    searchable = " ".join(
        str(tool_input.get(key, "")).strip().lower()
        for key in (
            "query",
            "pattern",
            "search",
            "text",
            "path",
            "file_path",
            "search_path",
        )
        if str(tool_input.get(key, "")).strip()
    )
    if name in {"view_file", "read"}:
        return bool(
            expected_file and path_text and path_text.endswith(expected_file)
        )
    if name in {"grep_search", "search", "grep", "rg"}:
        return bool(
            (expected_file and expected_file in searchable)
            or (expected_verification and expected_verification in searchable)
        )
    return False


def _classify_validated_target_tool_use(
    tool_uses: list[dict[str, Any]], expected_fields: dict[str, str]
) -> dict[str, int]:
    if not tool_uses:
        return {}
    if any(
        _tool_directly_reopens_expected_target(block, expected_fields)
        for block in tool_uses
    ):
        return {
            "validated_target_reacquired": 1,
            "validated_target_exact_reacquired": 1,
        }
    return {"validated_target_reconfirmation_attempt": 1}
