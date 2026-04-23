from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ._definitions import DEFAULT_MULTI_TURN_PROMPTS
from ._models import BenchmarkDefinition
from ._utils import _content_text


def load_fixture_messages(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record.get("messages"), list):
            records.extend(record["messages"])
        elif "role" in record and "content" in record:
            records.append(record)
    return records


def normalize_fixture_messages(messages: list[dict[str, Any]], followup_prompt: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if role == "tool_result":
            tool_id = str(msg.get("tool_use_id", "")).strip() or "unknown"
            normalized.append(
                {
                    "role": "user",
                    "content": f"Tool result ({tool_id}): {_content_text(msg.get('content', ''))}",
                }
            )
            continue

        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                normalized.append({"role": role or "user", "content": content})
            continue
        if not isinstance(content, list):
            continue

        new_content: list[dict[str, str]] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append({"type": "text", "text": str(block)})
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    new_content.append({"type": "text", "text": text})
            elif block_type in ("tool_use", "tool_result"):
                # Preserve structured blocks for Tok runtime analysis but stringify for baseline simplicity in tests
                if role == "assistant" and block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    new_content.append({"type": "text", "text": f"Tool use ({tool_name})"})
                new_content.append(block)

        if new_content:
            # Flatten to string if possible for broader provider compatibility (e.g. Bedrock)
            if all(block.get("type") == "text" for block in new_content):
                text_content = "\n".join(block.get("text", "") for block in new_content).strip()
                normalized.append({"role": role or "user", "content": text_content})
            else:
                normalized.append({"role": role or "user", "content": new_content})

    # Flatten the final followup_prompt as well
    if isinstance(followup_prompt, str):
        normalized.append({"role": "user", "content": followup_prompt})
    else:
        normalized.append({"role": "user", "content": followup_prompt})
    return normalized


def normalize_fixture_messages_for_bridge(messages: list[dict[str, Any]], followup_prompt: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    pending_tool_use_ids: set[str] = set()
    for msg in messages:
        role = str(msg.get("role", "")).strip() or "user"
        if role == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                tool_use_ids = [
                    str(block.get("id", "")).strip()
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ]
                # Track the most recent tool_use batch so we can keep strict pairing
                # and downgrade orphan/replayed tool_results to plain text.
                pending_tool_use_ids = {tool_id for tool_id in tool_use_ids if tool_id}
        if role == "tool_result":
            tool_id = str(msg.get("tool_use_id", "")).strip()
            tool_content = copy.deepcopy(msg.get("content", ""))
            if tool_id and tool_id in pending_tool_use_ids:
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_content,
                }
                normalized.append({"role": "user", "content": [tool_result_block]})
                pending_tool_use_ids.discard(tool_id)
            else:
                # Orphan/replayed tool_result: stringify to avoid invalid tool history upstream.
                normalized.append(
                    {
                        "role": "user",
                        "content": f"Tool result ({tool_id or 'unknown'}): {_content_text(tool_content)}",
                    }
                )
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            normalized.append({"role": role, "content": copy.deepcopy(content)})
            continue
        if isinstance(content, str):
            if content.strip():
                normalized.append({"role": role, "content": content})
            continue
        if content is not None:
            normalized.append({"role": role, "content": str(content)})

    normalized.append({"role": "user", "content": followup_prompt})
    return normalized


def _flatten_message_content_for_provider(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role", "")).strip() or "user"
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if not isinstance(content, list):
        return {"role": role, "content": _content_text(content)}

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                parts.append(text)
            continue

        block_type = str(block.get("type", "")).strip()
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)
            continue
        if block_type == "tool_use":
            tool_name = str(block.get("name", "unknown")).strip() or "unknown"
            tool_input = _content_text(block.get("input", "")).strip()
            parts.append(f"Tool use ({tool_name})" + (f": {tool_input}" if tool_input else ""))
            continue
        if block_type == "tool_result":
            tool_id = str(block.get("tool_use_id", "")).strip() or "unknown"
            tool_content = _content_text(block.get("content", "")).strip()
            parts.append(f"Tool result ({tool_id})" + (f": {tool_content}" if tool_content else ""))
            continue
        if block_type in {"thinking", "redacted_thinking"}:
            block_text = _content_text(block.get("thinking", block.get("data", ""))).strip()
            if block_text:
                parts.append(block_text)
            continue
        block_text = _content_text(block).strip()
        if block_text:
            parts.append(block_text)

    return {"role": role, "content": "\n".join(parts).strip()}


def _provider_safe_chat_messages(messages: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    if provider.lower() == "anthropic":
        return copy.deepcopy(messages)
    return [_flatten_message_content_for_provider(message) for message in messages]


def _turn_prompts(definition: BenchmarkDefinition, turns: int) -> list[str]:
    if definition.prompt_sequence:
        prompts = list(definition.prompt_sequence[:turns])
        while len(prompts) < turns:
            prompts.append(definition.followup_prompt)
        if turns > 0:
            prompts[-1] = definition.followup_prompt
        return prompts
    prompts = list(DEFAULT_MULTI_TURN_PROMPTS[:turns])
    while len(prompts) < turns:
        prompts.append(definition.followup_prompt)
    if turns > 0:
        prompts[-1] = definition.followup_prompt
    return prompts


def _chunk_messages(messages: list[dict[str, Any]], turns: int) -> list[list[dict[str, Any]]]:
    if turns <= 1:
        return [messages]
    if not messages:
        return [[] for _ in range(turns)]

    def _is_tool_result_message(message: dict[str, Any]) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip().startswith("Tool result (")
        if isinstance(content, list):
            blocks = [block for block in content if isinstance(block, dict)]
            return bool(blocks) and all(block.get("type") == "tool_result" for block in blocks)
        return False

    def _is_user_authored_message(message: dict[str, Any]) -> bool:
        return bool(message.get("role") == "user" and not _is_tool_result_message(message))

    units: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if _is_user_authored_message(message) and current:
            units.append(current)
            current = []
        current.append(message)
    if current:
        units.append(current)

    if not units:
        return [messages] + [[] for _ in range(turns - 1)]

    chunks: list[list[dict[str, Any]]] = []
    previous_target = 0
    for idx in range(turns):
        target = round(((idx + 1) * len(units)) / turns)
        target = max(previous_target, min(target, len(units)))
        chunk_units = units[previous_target:target]
        chunk: list[dict[str, Any]] = []
        for unit in chunk_units:
            chunk.extend(unit)
        chunks.append(chunk)
        previous_target = target

    if previous_target < len(units):
        chunks[-1].extend(message for unit in units[previous_target:] for message in unit)
    return chunks
