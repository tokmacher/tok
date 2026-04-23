from __future__ import annotations

import json
from typing import Any

from ._fixtures import _flatten_message_content_for_provider
from ._utils import _content_text

_OPENAI_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "view_file": {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "Read the contents of a file, optionally a line range",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "start": {"type": "integer", "description": "Start line number (1-based, inclusive)"},
                    "end": {"type": "integer", "description": "End line number (inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    "grep_search": {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents using a regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
    "list_dir": {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["path"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing old text with new text",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    "run_tests": {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a pytest command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The pytest command to run"},
                },
                "required": ["command"],
            },
        },
    },
    "git_diff": {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff of changes",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to diff"},
                },
                "required": [],
            },
        },
    },
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command (only pytest commands are allowed in this benchmark)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                },
                "required": ["command"],
            },
        },
    },
}


def _build_openai_tools_param(allowed_tools: tuple[str, ...]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool_name in allowed_tools:
        schema = _OPENAI_TOOL_SCHEMAS.get(tool_name)
        if schema:
            tools.append(schema)
    return tools


def _convert_openai_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for tc in raw_tool_calls:
        tool_id = str(getattr(tc, "id", "") or "")
        func = getattr(tc, "function", None)
        name = str(getattr(func, "name", "") or "") if func else ""
        arguments_raw = str(getattr(func, "arguments", "{}") or "{}") if func else "{}"
        try:
            tool_input = json.loads(arguments_raw)
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": tool_input,
            }
        )
    return blocks


def _adapt_tool_results_for_openai(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    pending_tool_call_ids: set[str] = set()

    def _stringify_tool_result_block(block: dict[str, Any]) -> str:
        tool_use_id = str(block.get("tool_use_id", "")).strip() or "unknown"
        content_text = _content_text(block.get("content", "")).strip()
        return f"Tool result ({tool_use_id}): {content_text}" if content_text else f"Tool result ({tool_use_id})"

    def _assistant_tool_call_message(content_blocks: list[dict[str, Any]]) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            block_type = str(block.get("type", "")).strip()
            if block_type == "tool_use":
                call_id = str(block.get("id", "")).strip()
                tool_name = str(block.get("name", "")).strip() or "unknown"
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    tool_input = {"value": tool_input}
                if call_id:
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input, ensure_ascii=False),
                            },
                        }
                    )
                else:
                    text_parts.append(f"Tool use ({tool_name}): {_content_text(tool_input)}")
                continue
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    text_parts.append(text)
                continue
            text = _content_text(block).strip()
            if text:
                text_parts.append(text)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts).strip(),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            assistant_blocks = [block for block in content if isinstance(block, dict)]
            assistant_message = _assistant_tool_call_message(assistant_blocks)
            tool_calls = assistant_message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                pending_tool_call_ids = {
                    str(call.get("id", "")).strip() for call in tool_calls if str(call.get("id", "")).strip()
                }
            else:
                pending_tool_call_ids = set()
            adapted.append(assistant_message)
            continue

        if role == "assistant":
            pending_tool_call_ids = set()
            adapted.append(_flatten_message_content_for_provider(msg))
            continue

        if role == "user" and isinstance(content, list):
            tool_results: list[dict[str, Any]] = []
            non_tool_result_blocks: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    non_tool_result_blocks.append({"type": "text", "text": str(block)})
                    continue
                if block.get("type") == "tool_result":
                    tool_results.append(block)
                else:
                    non_tool_result_blocks.append(block)

            if tool_results:
                for block in tool_results:
                    tool_use_id = str(block.get("tool_use_id", "")).strip()
                    if tool_use_id and tool_use_id in pending_tool_call_ids:
                        adapted.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": _content_text(block.get("content", "")),
                            }
                        )
                        pending_tool_call_ids.discard(tool_use_id)
                    else:
                        adapted.append({"role": "user", "content": _stringify_tool_result_block(block)})
                if non_tool_result_blocks:
                    adapted.append(
                        _flatten_message_content_for_provider({"role": "user", "content": non_tool_result_blocks})
                    )
                continue

        if pending_tool_call_ids:
            pending_tool_call_ids = set()
        adapted.append(msg)

    return adapted


def _detect_tool_protocol_retry_reason(error: Exception) -> str | None:
    message = str(error).lower()
    if "400" not in message and "invalid_request_error" not in message:
        return None
    if "no tool call found for function call output with call_id" in message:
        return "missing_tool_call_for_call_id"
    if "unexpected tool_use_id found in tool_result blocks" in message:
        return "unexpected_tool_use_id"
    if "must have corresponding tool_use in previous message" in message:
        return "tool_result_without_previous_tool_use"
    if "tool_call_id" in message and "preceding message" in message:
        return "tool_call_id_pairing_error"
    if "messages with role 'tool'" in message and "tool_calls" in message:
        return "orphan_tool_role_message"
    return None
