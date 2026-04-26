"""Tool repeat tracking and cache-sensitive result processing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from tok.compression import _strip_harness_injections, text_of
from tok.runtime.config import TOOL_DENSITY_THRESHOLD, TOOL_VOLUME_HEAVY_BYTES
from tok.runtime.repeat_targets import (
    SEARCH_LIKE_TOOLS,
    extract_shell_file_read_path,
)

from ._tool_context import logical_target_key_from_context

SignalBump = Callable[[str, int], None]


def _count_tool_density(messages: list[dict[str, Any]]) -> tuple[int, int]:
    tool_uses = 0
    tool_results = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool_result":
            tool_results += 1
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_uses += 1
    return tool_uses, tool_results


def _count_heavy_results(messages: list[dict[str, Any]]) -> int:
    heavy_results = 0
    for msg in messages:
        if msg.get("role") == "tool_result":
            raw = msg.get("content", "")
            if isinstance(raw, str) and len(raw) > TOOL_VOLUME_HEAVY_BYTES:
                heavy_results += 1
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                raw = block.get("content", "")
                if isinstance(raw, str) and len(raw) > TOOL_VOLUME_HEAVY_BYTES:
                    heavy_results += 1
    return heavy_results


def _should_skip_history_rewrite(
    messages: list[dict[str, Any]],
    normalized_tool_events: list[Any],
    *,
    tool_compatible: bool,
) -> tuple[bool, str]:
    tool_uses, tool_results = _count_tool_density(messages)
    total = len(messages) or 1
    density = tool_results / total

    file_like_reads = sum(1 for event in normalized_tool_events if event.compressibility_class == "file_read")
    command_like = sum(1 for event in normalized_tool_events if event.compressibility_class == "command")
    heavy_results = _count_heavy_results(messages)

    if tool_compatible:
        if density >= TOOL_DENSITY_THRESHOLD and tool_results >= 8:
            return False, "tool_density_high"
        if tool_uses >= 10:
            return False, "tool_use_count_high"
        if file_like_reads >= 6 and command_like >= 5:
            return False, "file_and_command_heavy"
        if heavy_results >= 6:
            return True, "tool_volume_heavy"
        return False, ""
    if density >= TOOL_DENSITY_THRESHOLD and tool_results >= 8:
        return False, "tool_density_high"
    if tool_uses >= 10:
        return False, "tool_use_count_high"
    if file_like_reads >= 6 and command_like >= 5:
        return False, "file_and_command_heavy"
    if heavy_results >= 4:
        return True, "tool_volume_heavy"
    return False, ""


def _iter_tool_results(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") == "tool_result":
            tool_id = str(msg.get("tool_use_id", "")).strip()
            if tool_id:
                results.append((tool_id, text_of(msg.get("content", ""))))
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_id = str(block.get("tool_use_id", "")).strip()
                if tool_id:
                    results.append((tool_id, text_of(block.get("content", ""))))
    return results


def _extract_tool_input_params(
    tool_input: dict[str, Any],
) -> tuple[str, str, str]:
    path = str(
        tool_input.get("path")
        or tool_input.get("file_path")
        or tool_input.get("AbsolutePath")
        or tool_input.get("TargetFile")
        or ""
    ).strip()
    command = str(tool_input.get("command") or tool_input.get("cmd") or "").strip()
    query = str(
        tool_input.get("query") or tool_input.get("pattern") or tool_input.get("search") or tool_input.get("text") or ""
    ).strip()
    return path, command, query


def _track_file_read_repeats(
    effective_path: str,
    logical_target: str,
    family: str,
    _tool_name: str,
    is_shell_file_read: bool,
    tool_id: str,
    file_reads_seen_raw: dict[str, int],
    file_reads_seen_logical: dict[str, int],
    repeat_file_read_ids: list[str],
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None,
    bump: SignalBump,
    has_bypass: bool = False,
) -> None:
    if family != "file_read" or logical_target == "path-missing":
        return
    file_reads_seen_raw[effective_path] = file_reads_seen_raw.get(effective_path, 0) + 1
    file_reads_seen_logical[logical_target] = file_reads_seen_logical.get(logical_target, 0) + 1
    if is_shell_file_read:
        bump("shell_file_read_normalized", 1)
    if file_reads_seen_logical[logical_target] > 1:
        if has_bypass:
            # Bypass flag prevents repeat penalty
            bump("bypass_reacquire", 1)
        elif tool_id and result_cache is not None:
            repeat_file_read_ids.append(tool_id)
        else:
            bump("repeat_file_read", 1)


def _track_search_repeats(
    query: str,
    logical_target: str,
    tool_name: str,
    tool_id: str,
    searches_seen_logical: dict[str, int],
    repeat_search_ids: list[str],
    bump: SignalBump,
) -> None:
    if tool_name not in SEARCH_LIKE_TOOLS or not query:
        return
    searches_seen_logical[logical_target] = searches_seen_logical.get(logical_target, 0) + 1
    if searches_seen_logical[logical_target] > 1:
        if tool_id:
            repeat_search_ids.append(tool_id)
        else:
            bump("repeat_search", 1)


def _track_command_repeats(
    logical_target: str,
    family: str,
    tool_id: str,
    commands_seen_logical: dict[str, int],
    repeat_command_ids: list[str],
    repeated_tool_targets: set[tuple[str, str]],
    _bump: SignalBump,
) -> None:
    if family != "command" or not logical_target:
        return
    commands_seen_logical[logical_target] = commands_seen_logical.get(logical_target, 0) + 1
    if commands_seen_logical[logical_target] > 1:
        if tool_id:
            repeat_command_ids.append(tool_id)
        repeated_tool_targets.add((family, logical_target))


def _detect_repeated_tool_calls(
    tool_uses: list[dict[str, Any]],
    family: str,
    logical_target: str,
    repeated_tool_targets: set[tuple[str, str]],
    bump: SignalBump,
) -> None:
    if family not in {"search", "command"} or not logical_target:
        return
    target_key = (family, logical_target)
    if target_key in repeated_tool_targets:
        bump("repeated_tool_call", 1)
        bump("error_detected", 1)
        return
    count = sum(
        1
        for tool_use in tool_uses
        if isinstance(tool_use, dict)
        and logical_target_key_from_context(
            str(tool_use.get("name", "")).lower(),
            path=str(
                (tool_use.get("input") or {}).get("path")
                or (tool_use.get("input") or {}).get("file_path")
                or (tool_use.get("input") or {}).get("AbsolutePath")
                or (tool_use.get("input") or {}).get("TargetFile")
                or ""
            ).strip()
            or None,
            query=str(
                (tool_use.get("input") or {}).get("query")
                or (tool_use.get("input") or {}).get("pattern")
                or (tool_use.get("input") or {}).get("search")
                or (tool_use.get("input") or {}).get("text")
                or ""
            ).strip()
            or None,
            command=str(
                (tool_use.get("input") or {}).get("command") or (tool_use.get("input") or {}).get("cmd") or ""
            ).strip()
            or None,
        )[:2]
        == target_key
    )
    if count > 1:
        bump("repeated_tool_call", 1)
        bump("error_detected", 1)


def _make_cache_key(tool_name: str, context: dict[str, Any]) -> str:
    args_str = json.dumps(context.get("args", {}), sort_keys=True)
    raw_key = f"{tool_name}:{args_str}"
    return hashlib.sha256(raw_key.encode()).hexdigest()[:12]


def _is_cached_hit(
    tool_name: str,
    context: dict[str, Any],
    raw_content: str,
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None,
) -> bool:
    if result_cache is None:
        return False
    cache_key = _make_cache_key(tool_name, context)
    if cache_key not in result_cache:
        return False
    cached_entry = result_cache[cache_key]
    if isinstance(cached_entry, dict):
        cached_hash = cached_entry.get("hash", "")
    elif isinstance(cached_entry, tuple | list) and cached_entry:
        cached_hash = cached_entry[0]
    else:
        cached_hash = ""
    current_hash = hashlib.sha256(_strip_harness_injections(raw_content).encode()).hexdigest()[:8]
    return current_hash == cached_hash


def _process_repeat_file_results(
    effective_path: str,
    logical_target: str,
    family: str,
    tool_name: str,
    context: dict[str, Any],
    result_text: str,
    is_repeat_file: bool,
    file_reads_seen_logical: dict[str, int],
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None,
    bump: SignalBump,
    count_tokens: Callable[[str], int],
) -> None:
    if family != "file_read":
        return
    if not effective_path or file_reads_seen_logical.get(logical_target, 0) <= 1:
        return
    if is_repeat_file:
        if result_cache is not None and _is_cached_hit(tool_name, context, result_text, result_cache):
            bump("cached_file_read", 1)
        else:
            bump("repeat_file_read", 1)
            reacquired = max(0, count_tokens(result_text))
            if reacquired:
                bump("reacquisition_cost_tokens", reacquired)


def _process_repeat_search_results(
    logical_target: str,
    tool_name: str,
    context: dict[str, Any],
    result_text: str,
    query: str,
    is_repeat_search: bool,
    searches_seen_logical: dict[str, int],
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None,
    bump: SignalBump,
    count_tokens: Callable[[str], int],
) -> None:
    if tool_name not in SEARCH_LIKE_TOOLS or not query:
        return
    if searches_seen_logical.get(logical_target, 0) <= 1:
        return
    if is_repeat_search:
        if result_cache is not None and _is_cached_hit(tool_name, context, result_text, result_cache):
            bump("cached_search", 1)
        else:
            bump("repeat_search", 1)
            reacquired = max(0, count_tokens(result_text))
            if reacquired:
                bump("reacquisition_cost_tokens", reacquired)


def _process_repeat_command_results(
    logical_target: str,
    family: str,
    result_text: str,
    is_repeat_command: bool,
    commands_seen_logical: dict[str, int],
    command_result_state: dict[str, tuple[str, bool]],
    bump: SignalBump,
) -> None:
    if family != "command":
        return
    text = str(result_text or "")
    digest = hashlib.sha256(text.encode()).hexdigest() if text else ""
    lowered = text.lower()
    current_success = bool(text.strip()) and not any(
        token in lowered for token in ("error", "failed", "traceback", "exception")
    )
    previous_state = command_result_state.get(logical_target)
    command_result_state[logical_target] = (digest, current_success)
    if commands_seen_logical.get(logical_target, 0) <= 1:
        return
    if is_repeat_command:
        bump("repeat_command", 1)
        if (
            previous_state
            and previous_state[1]
            and current_success
            and previous_state[0]
            and previous_state[0] == digest
        ):
            bump("repeat_command_stable_no_change", 1)
            bump("repeated_tool_call", 1)


def _process_cached_tool_results(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]],
    repeat_file_read_ids: list[str],
    repeat_search_ids: list[str],
    repeat_command_ids: list[str],
    file_reads_seen_logical: dict[str, int],
    searches_seen_logical: dict[str, int],
    commands_seen_logical: dict[str, int],
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None,
    bump: SignalBump,
    count_tokens: Callable[[str], int],
) -> None:
    command_result_state: dict[str, tuple[str, bool]] = {}
    for tool_id, result_text in _iter_tool_results(messages):
        context = tool_use_id_to_context.get(tool_id)
        if not context:
            continue
        tool_name = str(context.get("name", "")).lower()
        family, logical_target, _ = logical_target_key_from_context(
            tool_name,
            path=str(context.get("path") or "").strip() or None,
            query=str(context.get("query") or "").strip() or None,
            command=str(
                (context.get("args") or {}).get("command") or (context.get("args") or {}).get("cmd") or ""
            ).strip()
            or None,
        )
        effective_path = (
            str(context.get("path") or "").strip()
            or extract_shell_file_read_path(
                str((context.get("args") or {}).get("command") or (context.get("args") or {}).get("cmd") or "").strip()
            )
            or ""
        )
        _process_repeat_file_results(
            effective_path,
            logical_target,
            family,
            tool_name,
            context,
            result_text,
            tool_id in repeat_file_read_ids,
            file_reads_seen_logical,
            result_cache,
            bump,
            count_tokens,
        )
        _process_repeat_search_results(
            logical_target,
            tool_name,
            context,
            result_text,
            str(context.get("query") or "").strip(),
            tool_id in repeat_search_ids,
            searches_seen_logical,
            result_cache,
            bump,
            count_tokens,
        )
        _process_repeat_command_results(
            logical_target,
            family,
            result_text,
            tool_id in repeat_command_ids,
            commands_seen_logical,
            command_result_state,
            bump,
        )


__all__ = [
    "_count_tool_density",
    "_detect_repeated_tool_calls",
    "_extract_tool_input_params",
    "_process_cached_tool_results",
    "_should_skip_history_rewrite",
    "_track_command_repeats",
    "_track_file_read_repeats",
    "_track_search_repeats",
]
