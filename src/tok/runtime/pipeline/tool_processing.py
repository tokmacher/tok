"""Tool event processing and behavior signal collection."""

import hashlib
import json
from typing import Any, cast, Literal
from collections.abc import Callable

from ...compression import FILE_LIKE_TOOLS, text_of
from ..types import NormalizedToolEvent
from ..repeat_targets import (
    SEARCH_LIKE_TOOLS,
    display_target_label,
    extract_shell_file_read_path,
    logical_target_identity,
    normalize_tool_family,
)
from ..config import TOOL_DENSITY_THRESHOLD, TOOL_VOLUME_HEAVY_BYTES

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text, disallowed_special=()))

except ImportError:

    def count_tokens(text: str) -> int:
        return len(text) // 4


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
        "raise ",  # captures "raise SomeError" lines from grep/search results
    )
)
_HARD_BLOCKER_PHRASES_SET = frozenset(("blocked on ", "blocked by "))


def logical_target_key_from_context(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
) -> tuple[str, str, str]:
    family, logical_target = logical_target_identity(
        tool_name, path=path, query=query, command=command
    )
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
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            if not tool_id or not isinstance(tool_input, dict):
                continue
            path = (
                tool_input.get("path")
                or tool_input.get("file_path")
                or tool_input.get("AbsolutePath")
                or tool_input.get("TargetFile")
            )
            query = (
                tool_input.get("query")
                or tool_input.get("pattern")
                or tool_input.get("search")
                or tool_input.get("text")
            )
            result[tool_id] = {
                "name": tool_name,
                "args": tool_input,
                "path": str(path).strip() if path else None,
                "query": str(query).strip() if query else None,
            }
    return result


def normalize_tool_events(
    messages: list[dict[str, Any]],
) -> list[NormalizedToolEvent]:
    """Normalize assistant tool_use blocks into runtime-level events."""
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
            path = (
                str(
                    tool_input.get("path")
                    or tool_input.get("file_path")
                    or tool_input.get("AbsolutePath")
                    or tool_input.get("TargetFile")
                    or ""
                ).strip()
                or None
            )
            command = (
                str(
                    tool_input.get("command") or tool_input.get("cmd") or ""
                ).strip()
                or None
            )
            query = (
                str(
                    tool_input.get("query")
                    or tool_input.get("pattern")
                    or tool_input.get("search")
                    or tool_input.get("text")
                    or ""
                ).strip()
                or None
            )
            compressibility_class = normalize_tool_family(
                tool_name, query=query, command=command
            )
            if compressibility_class not in {"file_read", "search", "command"}:
                compressibility_class = "tool_result"
            compressibility_class = cast(
                Literal[
                    "raw", "file_read", "search", "command", "tool_result"
                ],
                compressibility_class,
            )
            fidelity_requirement = "high" if path or command else "default"
            events.append(
                NormalizedToolEvent(
                    id=str(block.get("id", "")),
                    name=tool_name,
                    args=tool_input,
                    path=path,
                    command=command,
                    query=query,
                    compressibility_class=compressibility_class,
                    fidelity_requirement=fidelity_requirement,
                )
            )
    return events


def _count_tool_density(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Count tool uses and tool results in message history."""
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
    """Count tool results whose payload exceeds TOOL_VOLUME_HEAVY_BYTES."""
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
    normalized_tool_events: list[NormalizedToolEvent],
    *,
    tool_compatible: bool,
) -> tuple[bool, str]:
    """Return (should_skip, reason) using deterministic density/volume thresholds.

    Reason is a short label emitted into behavior signals so CI can audit skips.
    """
    tool_uses, tool_results = _count_tool_density(messages)
    total = len(messages) or 1
    density = tool_results / total

    file_like_reads = sum(
        1
        for event in normalized_tool_events
        if event.compressibility_class == "file_read"
    )
    command_like = sum(
        1
        for event in normalized_tool_events
        if event.compressibility_class == "command"
    )

    # Volume: count tool results whose payload exceeds TOOL_VOLUME_HEAVY_BYTES
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
    else:
        if density >= TOOL_DENSITY_THRESHOLD and tool_results >= 8:
            return False, "tool_density_high"
        if tool_uses >= 10:
            return False, "tool_use_count_high"
        if file_like_reads >= 6 and command_like >= 5:
            return False, "file_and_command_heavy"
        if heavy_results >= 4:
            return False, "tool_volume_heavy"
        return False, ""


def _iter_tool_results(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Yield (tool_use_id, result_text) pairs from message history."""
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
                    results.append(
                        (tool_id, text_of(block.get("content", "")))
                    )
    return results


def _detect_blocker_rediscovery(
    messages: list[dict[str, Any]], blocker_phrases_seen: dict[str, int]
) -> None:
    """Detect blocker phrase rediscovery across messages."""
    for msg in messages:
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            is_hard = any(
                phrase in lowered for phrase in _HARD_BLOCKER_PHRASES_SET
            )
            is_transient = any(
                phrase in lowered for phrase in _TRANSIENT_ERROR_PHRASES_SET
            )

            if is_hard or is_transient:
                blocker_phrases_seen[lowered] = (
                    blocker_phrases_seen.get(lowered, 0) + 1
                )


def _extract_tool_input_params(
    tool_input: dict[str, Any],
) -> tuple[str, str, str]:
    """Extract path, command, and query from tool input dict."""
    path = str(
        tool_input.get("path")
        or tool_input.get("file_path")
        or tool_input.get("AbsolutePath")
        or tool_input.get("TargetFile")
        or ""
    ).strip()
    command = str(
        tool_input.get("command") or tool_input.get("cmd") or ""
    ).strip()
    query = str(
        tool_input.get("query")
        or tool_input.get("pattern")
        or tool_input.get("search")
        or tool_input.get("text")
        or ""
    ).strip()
    return path, command, query


def _detect_shell_workarounds(
    tool_name: str, command: str, bump: Callable[[str], None]
) -> None:
    """Detect shell command workarounds (python -c, stderr redirection)."""
    if not (
        tool_name in {"bash", "sh", "run_terminal", "computer"} and command
    ):
        return

    lowered = command.lower()
    if "python -c" in lowered or "python3 -c" in lowered:
        bump("python_c_workaround")
    if "/dev/stderr" in lowered or ">&2" in lowered or "2>&1" in lowered:
        bump("stderr_workaround")


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
    result_cache: dict[str, tuple[str, str]] | None,
    bump: Callable[[str], None],
) -> None:
    """Track file read operations and detect repeats."""
    if family != "file_read" or logical_target == "path-missing":
        return

    file_reads_seen_raw[effective_path] = (
        file_reads_seen_raw.get(effective_path, 0) + 1
    )
    file_reads_seen_logical[logical_target] = (
        file_reads_seen_logical.get(logical_target, 0) + 1
    )

    if is_shell_file_read:
        bump("shell_file_read_normalized")

    if file_reads_seen_logical[logical_target] > 1:
        if tool_id and result_cache is not None:
            repeat_file_read_ids.append(tool_id)
        else:
            bump("repeat_file_read")


def _track_search_repeats(
    query: str,
    logical_target: str,
    tool_name: str,
    tool_id: str,
    searches_seen_logical: dict[str, int],
    repeat_search_ids: list[str],
    bump: Callable[[str], None],
) -> None:
    """Track search operations and detect repeats."""
    if tool_name not in SEARCH_LIKE_TOOLS or not query:
        return

    searches_seen_logical[logical_target] = (
        searches_seen_logical.get(logical_target, 0) + 1
    )

    if searches_seen_logical[logical_target] > 1:
        if tool_id:
            repeat_search_ids.append(tool_id)
        else:
            bump("repeat_search")


def _track_command_repeats(
    logical_target: str,
    family: str,
    tool_id: str,
    commands_seen_logical: dict[str, int],
    repeat_command_ids: list[str],
    repeated_tool_targets: set[tuple[str, str]],
    _bump: Callable[[str], None],
) -> None:
    """Track command operations and detect repeats."""
    if family != "command" or not logical_target:
        return

    commands_seen_logical[logical_target] = (
        commands_seen_logical.get(logical_target, 0) + 1
    )

    if commands_seen_logical[logical_target] > 1:
        if tool_id:
            repeat_command_ids.append(tool_id)
        repeated_tool_targets.add((family, logical_target))


def _detect_repeated_tool_calls(
    tool_uses: list[dict[str, Any]],
    family: str,
    logical_target: str,
    repeated_tool_targets: set[tuple[str, str]],
    bump: Callable[[str], None],
) -> None:
    """Detect when the same logical target is used multiple times in a turn."""
    if family not in {"search", "command"} or not logical_target:
        return

    target_key = (family, logical_target)
    if target_key in repeated_tool_targets:
        bump("repeated_tool_call")
        bump("error_detected")
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
                (tool_use.get("input") or {}).get("command")
                or (tool_use.get("input") or {}).get("cmd")
                or ""
            ).strip()
            or None,
        )[:2]
        == target_key
    )
    if count > 1:
        bump("repeated_tool_call")
        bump("error_detected")


def _detect_prose_leaks(
    messages: list[dict[str, Any]], bump: Callable[[str], None]
) -> None:
    """Detect Tok protocol markers leaking into assistant text blocks."""
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


def _detect_error_patterns(
    messages: list[dict[str, Any]], bump: Callable[[str], None]
) -> None:
    """Detect error phrases in message content."""
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


def _make_cache_key(tool_name: str, context: dict[str, Any]) -> str:
    """Create a cache key for tool result caching."""
    args_str = json.dumps(context.get("args", {}), sort_keys=True)
    raw_key = f"{tool_name}:{args_str}"
    return hashlib.sha256(raw_key.encode()).hexdigest()[:12]


def _is_cached_hit(
    tool_name: str,
    context: dict[str, Any],
    raw_content: str,
    result_cache: dict[str, tuple[str, str]] | None,
) -> bool:
    """Check if a tool result is cached and matches."""
    if result_cache is None:
        return False
    cache_key = _make_cache_key(tool_name, context)
    if cache_key not in result_cache:
        return False
    cached_hash, _ = result_cache[cache_key]
    current_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:8]
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
    result_cache: dict[str, tuple[str, str]] | None,
    bump: Callable[[str], None],
) -> None:
    """Process repeat file read results and calculate reacquisition costs."""
    if family != "file_read":
        return
    if (
        not effective_path
        or file_reads_seen_logical.get(logical_target, 0) <= 1
    ):
        return

    if is_repeat_file:
        if result_cache is not None and _is_cached_hit(
            tool_name, context, result_text, result_cache
        ):
            bump("cached_file_read")
        else:
            bump("repeat_file_read")
            reacquired = max(0, count_tokens(result_text))
            if reacquired:
                bump("reacquisition_cost_tokens")


def _process_repeat_search_results(
    logical_target: str,
    tool_name: str,
    context: dict[str, Any],
    result_text: str,
    query: str,
    is_repeat_search: bool,
    searches_seen_logical: dict[str, int],
    result_cache: dict[str, tuple[str, str]] | None,
    bump: Callable[[str], None],
) -> None:
    """Process repeat search results and calculate reacquisition costs."""
    if tool_name not in SEARCH_LIKE_TOOLS or not query:
        return
    if searches_seen_logical.get(logical_target, 0) <= 1:
        return

    if is_repeat_search:
        if result_cache is not None and _is_cached_hit(
            tool_name, context, result_text, result_cache
        ):
            bump("cached_search")
        else:
            bump("repeat_search")
            reacquired = max(0, count_tokens(result_text))
            if reacquired:
                bump("reacquisition_cost_tokens")


def _process_repeat_command_results(
    logical_target: str,
    family: str,
    is_repeat_command: bool,
    commands_seen_logical: dict[str, int],
    bump: Callable[[str], None],
) -> None:
    """Process repeat command results."""
    if family != "command":
        return
    if commands_seen_logical.get(logical_target, 0) <= 1:
        return

    if is_repeat_command:
        bump("repeated_tool_call")


def collect_behavior_signals(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    result_cache: dict[str, tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Track patterns that indicate Tok is helping or being routed around."""
    signals: dict[str, int] = {}
    file_reads_seen_raw: dict[str, int] = {}
    file_reads_seen_logical: dict[str, int] = {}
    searches_seen_logical: dict[str, int] = {}
    commands_seen_logical: dict[str, int] = {}
    blocker_phrases_seen: dict[str, int] = {}
    repeat_file_read_ids: list[str] = []
    repeat_search_ids: list[str] = []
    repeat_command_ids: list[str] = []
    repeated_tool_targets: set[tuple[str, str]] = set()

    def bump(key: str, amount: int = 1) -> None:
        signals[key] = signals.get(key, 0) + amount

    # Detect blocker rediscovery
    _detect_blocker_rediscovery(messages, blocker_phrases_seen)
    _track_assistant_tool_usage(
        messages,
        file_reads_seen_raw,
        file_reads_seen_logical,
        searches_seen_logical,
        commands_seen_logical,
        repeat_file_read_ids,
        repeat_search_ids,
        repeat_command_ids,
        repeated_tool_targets,
        result_cache,
        bump,
    )

    if tool_use_id_to_context:
        _process_cached_tool_results(
            messages,
            tool_use_id_to_context,
            repeat_file_read_ids,
            repeat_search_ids,
            repeat_command_ids,
            file_reads_seen_logical,
            searches_seen_logical,
            commands_seen_logical,
            result_cache,
            bump,
        )

    # Detect prose leaks
    _detect_prose_leaks(messages, bump)

    # Detect error patterns
    _detect_error_patterns(messages, bump)

    # Final blocker rediscovery signals
    for count in blocker_phrases_seen.values():
        if count > 1:
            bump("blocker_rediscovery")
            break

    return signals


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
    result_cache: dict[str, tuple[str, str]] | None,
    bump: Callable[[str], None],
) -> None:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        tool_uses = [
            b
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
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
            shell_file_path = (
                extract_shell_file_read_path(command) if command else None
            )
            is_shell_file_read = bool(
                shell_file_path
                and family == "file_read"
                and tool_name not in FILE_LIKE_TOOLS
            )
            effective_path = path or shell_file_path or ""

            # Track file read repeats
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

            # Track search repeats
            _track_search_repeats(
                query,
                logical_target,
                tool_name,
                block.get("id", ""),
                searches_seen_logical,
                repeat_search_ids,
                bump,
            )

            # Track command repeats
            _track_command_repeats(
                logical_target,
                family,
                block.get("id", ""),
                commands_seen_logical,
                repeat_command_ids,
                repeated_tool_targets,
                bump,
            )

            # Detect shell workarounds
            _detect_shell_workarounds(tool_name, command, bump)

            # Detect repeated tool calls
            _detect_repeated_tool_calls(
                tool_uses, family, logical_target, repeated_tool_targets, bump
            )


def _process_cached_tool_results(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]],
    repeat_file_read_ids: list[str],
    repeat_search_ids: list[str],
    repeat_command_ids: list[str],
    file_reads_seen_logical: dict[str, int],
    searches_seen_logical: dict[str, int],
    commands_seen_logical: dict[str, int],
    result_cache: dict[str, tuple[str, str]] | None,
    bump: Callable[[str], None],
) -> None:
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
                (context.get("args") or {}).get("command")
                or (context.get("args") or {}).get("cmd")
                or ""
            ).strip()
            or None,
        )
        is_repeat_file = tool_id in repeat_file_read_ids
        is_repeat_search = tool_id in repeat_search_ids
        is_repeat_command = tool_id in repeat_command_ids
        effective_path = (
            str(context.get("path") or "").strip()
            or extract_shell_file_read_path(
                str(
                    (context.get("args") or {}).get("command")
                    or (context.get("args") or {}).get("cmd")
                    or ""
                ).strip()
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
            is_repeat_file,
            file_reads_seen_logical,
            result_cache,
            bump,
        )

        _process_repeat_search_results(
            logical_target,
            tool_name,
            context,
            result_text,
            str(context.get("query") or "").strip(),
            is_repeat_search,
            searches_seen_logical,
            result_cache,
            bump,
        )

        _process_repeat_command_results(
            logical_target,
            family,
            is_repeat_command,
            commands_seen_logical,
            bump,
        )


__all__ = [
    "build_tool_use_id_to_context",
    "normalize_tool_events",
    "collect_behavior_signals",
    "_count_tool_density",
    "_should_skip_history_rewrite",
    "count_tokens",
    "SEARCH_LIKE_TOOLS",
]
