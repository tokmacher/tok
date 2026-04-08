"""Tool context extraction and normalization helpers."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from tok.compression import FILE_LIKE_TOOLS
from tok.runtime.repeat_targets import (
    display_target_label,
    logical_target_identity,
    normalize_tool_family,
)
from tok.runtime.types import NormalizedToolEvent

logger = logging.getLogger("tok.runtime.tool_processing")


class ToolContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any]
    path: str | None = None
    query: str | None = None

    @field_validator("name")
    @classmethod
    def _name_must_not_be_blank(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            msg = "blank tool name"
            raise ValueError(msg)
        return normalized

    @field_validator("path", "query")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


def _is_supported_bypass_target(tool_name: str, tool_input: dict[str, Any]) -> bool:
    lowered = str(tool_name or "").lower()
    if lowered not in FILE_LIKE_TOOLS:
        return False
    return any(tool_input.get(key) for key in ("path", "file_path", "AbsolutePath", "TargetFile"))


def logical_target_key_from_context(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
) -> tuple[str, str, str]:
    family, logical_target = logical_target_identity(tool_name, path=path, query=query, command=command)
    return (
        family,
        logical_target,
        display_target_label(
            family,
            path=path,
            query=query,
            command=command,
            logical_target=logical_target,
        ),
    )


def _should_bypass_next_tool_use(block: dict[str, Any]) -> bool:
    """Check if a text block contains the bypass marker."""
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and "@tok_bypass_next_read" in str(block.get("text", ""))
    )


def _extract_tool_input_fields(
    tool_input: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Extract path and query fields from tool input."""
    path = (
        tool_input.get("path")
        or tool_input.get("file_path")
        or tool_input.get("AbsolutePath")
        or tool_input.get("TargetFile")
    )
    query = tool_input.get("query") or tool_input.get("pattern") or tool_input.get("search") or tool_input.get("text")
    return path, query


def _apply_bypass_marker(
    tool_input: dict[str, Any],
    tool_name: str,
    path: str | None,
    tool_id: str | None,
) -> tuple[dict[str, Any], bool]:
    """
    Apply bypass marker to tool_input if target is supported.

    Returns updated tool_input and invalid_bypass_marker flag.
    """
    if _is_supported_bypass_target(str(tool_name), tool_input):
        tool_input = dict(tool_input)
        tool_input["tok_bypass_cache"] = True
        logger.info(
            "tok_bypass_cache applied via marker | tool_id=%s tool=%s path=%s",
            tool_id,
            tool_name,
            str(path).strip() if path else "",
        )
        return tool_input, False
    logger.warning(
        "TOK_BYPASS_NEXT_READ_CONSUMED_BY_NON_READ_TOOL | marker ignored for unsupported tool | tool_id=%s tool=%s",
        tool_id,
        tool_name,
    )
    return tool_input, True


def _build_context_dict(
    tool_name: str | None,
    tool_input: dict[str, Any],
    path: str | None,
    query: str | None,
    invalid_bypass_marker: bool,
) -> dict[str, Any]:
    """Build and validate the context dictionary for a tool use."""
    context_payload = {
        "name": tool_name,
        "args": tool_input,
        "path": str(path).strip() if path else None,
        "query": str(query).strip() if query else None,
    }
    try:
        validated = ToolContextModel.model_validate(context_payload)
        context_dict = validated.model_dump()
    except ValidationError:
        context_dict = {
            "name": str(tool_name or "").strip(),
            "args": dict(tool_input) if isinstance(tool_input, dict) else {},
            "path": str(path).strip() if path else None,
            "query": str(query).strip() if query else None,
            "tool_context_validation_failed": True,
        }
    if invalid_bypass_marker and "tool_context_validation_failed" not in context_dict:
        context_dict["invalid_bypass_marker"] = True
    return context_dict


def _process_message_blocks(
    content: list[Any],
    result: dict[str, dict[str, Any]],
) -> None:
    """Process all blocks in a message for tool_use extraction."""
    bypass_next_tool_use = False
    invalid_bypass_index = 0
    for block in content:
        if _should_bypass_next_tool_use(block):
            bypass_next_tool_use = True
            logger.info("tok_bypass_next_read marker observed in assistant text")
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        invalid_bypass_index, bypass_consumed = _process_single_tool_use(
            block, result, bypass_next_tool_use, invalid_bypass_index
        )
        if bypass_consumed:
            bypass_next_tool_use = False


def _process_single_tool_use(
    block: dict[str, Any],
    result: dict[str, dict[str, Any]],
    bypass_next_tool_use: bool,
    invalid_bypass_index: int,
) -> tuple[int, bool]:
    """
    Process a single tool_use block and update result.

    Returns updated invalid_bypass_index and whether bypass marker was consumed.
    """
    tool_id = block.get("id", "")
    tool_name = block.get("name", "")
    tool_input = block.get("input", {})
    if not tool_id or not isinstance(tool_input, dict):
        if bypass_next_tool_use:
            invalid_bypass_index += 1
            result[f"__invalid_bypass_marker__:{invalid_bypass_index}"] = {"invalid_bypass_marker": True}
        return invalid_bypass_index, False

    path, query = _extract_tool_input_fields(tool_input)
    invalid_bypass_marker = False
    bypass_consumed = False
    if bypass_next_tool_use:
        bypass_consumed = _is_supported_bypass_target(str(tool_name), tool_input)
        tool_input, invalid_bypass_marker = _apply_bypass_marker(tool_input, str(tool_name), path, tool_id)
    result[tool_id] = _build_context_dict(tool_name, tool_input, path, query, invalid_bypass_marker)
    return invalid_bypass_index, bypass_consumed


def build_tool_use_id_to_context(
    messages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Walk assistant messages to build tool_use_id -> context map."""
    result: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        _process_message_blocks(content, result)
    return result


def collect_tool_context_validation_signals(
    tool_use_id_to_context: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Collect additive signals from validated tool-use contexts."""
    signals: dict[str, int] = {}
    for context in tool_use_id_to_context.values():
        if not isinstance(context, dict):
            continue
        if context.get("invalid_bypass_marker"):
            signals["invalid_bypass_marker_application"] = signals.get("invalid_bypass_marker_application", 0) + 1
        if context.get("tool_context_validation_failed"):
            signals["tool_context_validation_failed"] = signals.get("tool_context_validation_failed", 0) + 1
    return signals


def _extract_path(tool_input: dict[str, Any]) -> str | None:
    return (
        str(
            tool_input.get("path")
            or tool_input.get("file_path")
            or tool_input.get("AbsolutePath")
            or tool_input.get("TargetFile")
            or ""
        ).strip()
        or None
    )


def _extract_command(tool_input: dict[str, Any]) -> str | None:
    return str(tool_input.get("command") or tool_input.get("cmd") or "").strip() or None


def _extract_query(tool_input: dict[str, Any]) -> str | None:
    return (
        str(
            tool_input.get("query")
            or tool_input.get("pattern")
            or tool_input.get("search")
            or tool_input.get("text")
            or ""
        ).strip()
        or None
    )


def _extract_tok_mode_from_block(block: dict[str, Any]) -> object | None:
    """Extract TokMode from a text block containing runtime_session JSON."""
    import re

    from tok.runtime.smoothness.models import TokMode

    text = block.get("text", "")
    if "runtime_session" not in text:
        return None
    try:
        match = re.search(r'"current_tok_mode"\s*:\s*"([^"]+)"', text)
        if match:
            return TokMode(match.group(1))
    except Exception:
        pass
    return None


def _find_current_mode(messages: list[dict[str, Any]]) -> object | None:
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            mode = _extract_tok_mode_from_block(block)
            if mode:
                return mode
    return None


def _is_raw_mode(current_mode: object | None, path: str | None) -> bool:
    from tok.runtime.smoothness.models import TokMode

    return bool(
        current_mode
        in (
            TokMode.SMOOTH_MODE,
            TokMode.LOSSLESS_TASK_MODE,
        )
        and path
    )


def _build_normalized_event(
    block_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    path: str | None,
    command: str | None,
    query: str | None,
    current_mode: object | None,
) -> NormalizedToolEvent:
    compressibility_class: Literal["raw", "file_read", "search", "command", "tool_result"] = cast(
        "Literal['raw', 'file_read', 'search', 'command', 'tool_result']",
        normalize_tool_family(tool_name, query=query, command=command),
    )
    if compressibility_class not in {"file_read", "search", "command"}:
        compressibility_class = "tool_result"
    if _is_raw_mode(current_mode, path):
        compressibility_class = "raw"
    fidelity_requirement = "high" if path or command else "default"
    return NormalizedToolEvent(
        id=block_id,
        name=tool_name,
        args=tool_input,
        path=path,
        command=command,
        query=query,
        compressibility_class=compressibility_class,
        fidelity_requirement=fidelity_requirement,
    )


def normalize_tool_events(
    messages: list[dict[str, Any]],
) -> list[NormalizedToolEvent]:
    """Normalize assistant tool_use blocks into runtime-level events."""
    current_mode = _find_current_mode(messages)

    events: list[NormalizedToolEvent] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}
            tool_name = str(block.get("name", ""))
            path = _extract_path(tool_input)
            command = _extract_command(tool_input)
            query = _extract_query(tool_input)
            events.append(
                _build_normalized_event(
                    str(block.get("id", "")),
                    tool_name,
                    tool_input,
                    path,
                    command,
                    query,
                    current_mode,
                )
            )
    return events


__all__ = [
    "ToolContextModel",
    "build_tool_use_id_to_context",
    "collect_tool_context_validation_signals",
    "logical_target_key_from_context",
    "normalize_tool_events",
]
