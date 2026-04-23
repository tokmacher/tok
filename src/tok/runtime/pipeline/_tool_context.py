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


def _extract_tool_input_fields(
    tool_input: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Extract path and query fields from tool input."""
    path = (
        tool_input.get("path")
        or tool_input.get("file_path")
        or tool_input.get("AbsolutePath")
        or tool_input.get("TargetFile")
        or tool_input.get("SearchDirectory")  # find_by_name
        or tool_input.get("search_folder_absolute_uri")  # code_search
    )
    query = (
        tool_input.get("query")
        or tool_input.get("pattern")
        or tool_input.get("Pattern")  # find_by_name
        or tool_input.get("search")
        or tool_input.get("search_term")  # code_search
        or tool_input.get("text")
    )
    return path, query


def _build_context_dict(
    tool_name: str | None,
    tool_input: dict[str, Any],
    path: str | None,
    query: str | None,
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
        return validated.model_dump()
    except ValidationError:
        return {
            "name": str(tool_name or "").strip(),
            "args": dict(tool_input) if isinstance(tool_input, dict) else {},
            "path": str(path).strip() if path else None,
            "query": str(query).strip() if query else None,
            "tool_context_validation_failed": True,
        }


def _process_message_blocks(
    content: list[Any],
    result: dict[str, dict[str, Any]],
    session: Any = None,
) -> None:
    """Process all blocks in a message for tool_use extraction."""
    bypass_marker_pending = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            if "@tok_bypass_next_read" in text:
                bypass_marker_pending = True
            continue
        if block_type != "tool_use":
            continue
        tool_name = str(block.get("name", "")).lower()
        is_file_like = tool_name in FILE_LIKE_TOOLS
        # Only consume bypass marker on file-like tools
        # Non-file tools get the invalid marker but don't consume it
        apply_bypass = bypass_marker_pending
        if bypass_marker_pending and not is_file_like:
            # Non-file tool with pending marker - apply invalid marker but keep marker pending
            apply_bypass = True
        elif bypass_marker_pending and is_file_like:
            # File-like tool consumes the marker
            bypass_marker_pending = False
        _process_single_tool_use(block, result, apply_bypass, session)


def _process_single_tool_use(
    block: dict[str, Any],
    result: dict[str, dict[str, Any]],
    bypass_marker_pending: bool = False,
    session: Any = None,
) -> None:
    """Process a single tool_use block and update result."""
    tool_id = block.get("id", "")
    tool_name = block.get("name", "")
    tool_input = block.get("input", {})
    if not tool_id or not isinstance(tool_input, dict):
        return

    path, query = _extract_tool_input_fields(tool_input)
    context = _build_context_dict(tool_name, tool_input, path, query)

    # Add session to context if available
    if session is not None:
        context["session"] = session

    # Apply bypass marker if pending
    if bypass_marker_pending:
        normalized_tool_name = str(tool_name or "").lower()
        if normalized_tool_name in FILE_LIKE_TOOLS:
            # Valid: apply bypass to file-like tool
            context["args"] = dict(context.get("args", {}))
            context["args"]["tok_bypass_cache"] = True
        else:
            # Invalid: mark as invalid bypass application
            context["invalid_bypass_marker"] = True

    result[tool_id] = context


def build_tool_use_id_to_context(
    messages: list[dict[str, Any]],
    session: Any = None,
) -> dict[str, dict[str, Any]]:
    """Walk assistant messages to build tool_use_id -> context map."""
    result: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        _process_message_blocks(content, result, session)
    return result


def collect_tool_context_validation_signals(
    tool_use_id_to_context: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Collect additive signals from validated tool-use contexts."""
    signals: dict[str, int] = {}
    for context in tool_use_id_to_context.values():
        if not isinstance(context, dict):
            continue
        if context.get("tool_context_validation_failed"):
            signals["tool_context_validation_failed"] = signals.get("tool_context_validation_failed", 0) + 1
        if context.get("invalid_bypass_marker"):
            signals["invalid_bypass_marker_application"] = signals.get("invalid_bypass_marker_application", 0) + 1
    return signals


def _extract_path(tool_input: dict[str, Any]) -> str | None:
    return (
        str(
            tool_input.get("path")
            or tool_input.get("file_path")
            or tool_input.get("AbsolutePath")
            or tool_input.get("TargetFile")
            or tool_input.get("SearchDirectory")  # find_by_name
            or tool_input.get("search_folder_absolute_uri")  # code_search
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
            or tool_input.get("Pattern")  # find_by_name
            or tool_input.get("search")
            or tool_input.get("search_term")  # code_search
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
