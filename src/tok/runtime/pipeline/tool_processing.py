"""Tool event processing and behavior signal collection."""

from __future__ import annotations

from typing import Any

from ..repeat_targets import SEARCH_LIKE_TOOLS
from ._tool_context import (
    ToolContextModel,
    build_tool_use_id_to_context,
    collect_tool_context_validation_signals,
    logical_target_key_from_context,
    normalize_tool_events,
)
from ._tool_repeat_detection import (
    _count_tool_density,
    _process_cached_tool_results,
    _should_skip_history_rewrite,
)
from ._tool_signal_detection import (
    _HARD_BLOCKER_PHRASES_SET,
    _TRANSIENT_ERROR_PHRASES_SET,
    _detect_blocker_rediscovery,
    _detect_error_patterns,
    _detect_prose_leaks,
    _track_assistant_tool_usage,
)
from ._tool_repeat_detection import _iter_tool_results

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


def collect_behavior_signals(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    result_cache: dict[
        str, tuple[str, str, float] | tuple[str, str] | tuple[str]
    ]
    | None = None,
    suppress_reacquisition_once: bool = False,
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

    def bump_one(key: str) -> None:
        bump(key, 1)

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
            count_tokens,
        )

    _detect_prose_leaks(messages, bump_one)
    _detect_error_patterns(messages, bump_one)

    for count in blocker_phrases_seen.values():
        if count > 1:
            bump("blocker_rediscovery")
            break

    if suppress_reacquisition_once:
        suppressed = False
        for key in (
            "repeat_file_read",
            "repeat_search",
            "reacquisition_cost_tokens",
            "file_reacquisition_cost_tokens",
            "search_reacquisition_cost_tokens",
        ):
            if signals.pop(key, 0):
                suppressed = True
        if suppressed:
            bump("stream_recovery_reacquisition_suppressed")

    return signals


__all__ = [
    "ToolContextModel",
    "build_tool_use_id_to_context",
    "collect_behavior_signals",
    "collect_tool_context_validation_signals",
    "count_tokens",
    "logical_target_key_from_context",
    "normalize_tool_events",
    "_count_tool_density",
    "_HARD_BLOCKER_PHRASES_SET",
    "_TRANSIENT_ERROR_PHRASES_SET",
    "_iter_tool_results",
    "_should_skip_history_rewrite",
    "SEARCH_LIKE_TOOLS",
]
