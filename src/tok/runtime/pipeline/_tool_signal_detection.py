"""Behavior signal detectors for runtime tool usage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tok.compression import FILE_LIKE_TOOLS, text_of
from tok.runtime.repeat_targets import extract_shell_file_read_path

from ._tool_context import logical_target_key_from_context
from ._tool_repeat_detection import (
    _detect_repeated_tool_calls,
    _extract_tool_input_params,
    _track_command_repeats,
    _track_file_read_repeats,
    _track_search_repeats,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_TRANSIENT_ERROR_PHRASES_SET = frozenset(
    (
        "still failing",
        "fails with",
        "failing test",
        "importerror",
        "modulenotfounderror",
        "syntaxerror",
        "cannot import",
        "no module named",
        "dependency error",
        "error: command failed",
        "attributeerror",
        "typeerror",
        "raise ",
    )
)
_HARD_BLOCKER_PHRASES_SET = frozenset(("blocked on ", "blocked by "))


def _detect_blocker_rediscovery(messages: list[dict[str, Any]], blocker_phrases_seen: dict[str, int]) -> None:
    for msg in messages:
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            is_hard = any(phrase in lowered for phrase in _HARD_BLOCKER_PHRASES_SET)
            is_transient = any(phrase in lowered for phrase in _TRANSIENT_ERROR_PHRASES_SET)
            if is_hard or is_transient:
                blocker_phrases_seen[lowered] = blocker_phrases_seen.get(lowered, 0) + 1


def _detect_shell_workarounds(tool_name: str, command: str, bump: Callable[[str, int], None]) -> None:
    if not (tool_name in {"bash", "sh", "run_terminal", "computer"} and command):
        return
    lowered = command.lower()
    if "python -c" in lowered or "python3 -c" in lowered:
        bump("python_c_workaround", 1)
    if "/dev/stderr" in lowered or ">&2" in lowered or "2>&1" in lowered:
        bump("stderr_workaround", 1)


def _detect_prose_leaks(messages: list[dict[str, Any]], bump: Callable[[str], None]) -> None:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.startswith(">>>"):
                    text = "\n".join(text.split("\n")[1:])
                if ">>>" in text or "@msg" in text or "@field" in text:
                    bump("prose_leak_detected")
                    bump("semantic_drift")


def _detect_error_patterns(messages: list[dict[str, Any]], bump: Callable[[str], None]) -> None:
    error_patterns = [
        "permission denied",
        "no such file",
        "not found",
        "failed to",
    ]
    for msg in messages:
        text = text_of(msg.get("content", "")).lower()
        if any(p in text for p in error_patterns):
            bump("error_detected")


def _track_assistant_tool_usage(
    messages: list[dict[str, Any]],
    file_reads_seen_raw: dict[str, int],
    file_reads_seen_logical: dict[str, int],
    searches_seen_logical: dict[str, int],
    commands_seen_logical: dict[str, int],
    repeat_file_read_ids: list[str],
    repeat_search_ids: list[str],
    repeat_command_ids: list[str],
    repeated_tool_targets: set[tuple[str, str]],
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None,
    bump: Callable[[str, int], None],
) -> None:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = str(block.get("name", "")).lower()
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}
            path, command, query = _extract_tool_input_params(tool_input)
            family, logical_target, _ = logical_target_key_from_context(
                tool_name,
                path=path or None,
                query=query or None,
                command=command or None,
            )
            shell_file_path = extract_shell_file_read_path(command) if command else None
            is_shell_file_read = bool(shell_file_path and family == "file_read" and tool_name not in FILE_LIKE_TOOLS)
            effective_path = path or shell_file_path or ""
            _track_file_read_repeats(
                effective_path,
                logical_target,
                family,
                tool_name,
                is_shell_file_read,
                block.get("id", ""),
                file_reads_seen_raw,
                file_reads_seen_logical,
                repeat_file_read_ids,
                result_cache,
                bump,
            )
            _track_search_repeats(
                query,
                logical_target,
                tool_name,
                block.get("id", ""),
                searches_seen_logical,
                repeat_search_ids,
                bump,
            )
            _track_command_repeats(
                logical_target,
                family,
                block.get("id", ""),
                commands_seen_logical,
                repeat_command_ids,
                repeated_tool_targets,
                bump,
            )
            _detect_shell_workarounds(tool_name, command, bump)
            _detect_repeated_tool_calls(tool_uses, family, logical_target, repeated_tool_targets, bump)


__all__ = [
    "_HARD_BLOCKER_PHRASES_SET",
    "_TRANSIENT_ERROR_PHRASES_SET",
    "_detect_blocker_rediscovery",
    "_detect_error_patterns",
    "_detect_prose_leaks",
    "_track_assistant_tool_usage",
]
