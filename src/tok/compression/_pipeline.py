"""Compression pipeline facade and backward-compatible entrypoints."""

from __future__ import annotations

from typing import Any

from . import _history_pipeline as _history
from ._history_pipeline import (
    RECENT_WINDOW_THRESHOLD,
    _compress_git_log_impl,
    _detect_tool_content_type_impl,
    compress_history_impl,
    inject_system_additions_impl,
)
from ._history_pipeline import (
    compress_recent_window_impl as _compress_recent_window_impl,
)
from ._tool_result_codecs import (
    _compress_config_json,
    _compress_env_ps,
    _compress_file_read,
    _compress_git_diff,
    _compress_git_log,
    _compress_grep,
    _compress_grep_context,
    _compress_install,
    _compress_json_response,
    _compress_ls,
    _compress_pytest,
    _compress_repetitive,
    _compress_search_results,
    _compress_stack_traces,
    _detect_tool_content_type,
    _tighten_compressed_output,
    truncate_large_result,
)

__all__ = [
    "TOOL_COMPRESS_THRESHOLD",
    "_compress_config_json",
    "_compress_env_ps",
    "_compress_file_read",
    "_compress_git_diff",
    "_compress_git_log",
    "_compress_git_log_impl",
    "_compress_grep",
    "_compress_grep_context",
    "_compress_install",
    "_compress_json_response",
    "_compress_ls",
    "_compress_pytest",
    "_compress_repetitive",
    "_compress_search_results",
    "_compress_stack_traces",
    "_detect_tool_content_type",
    "_detect_tool_content_type_impl",
    "_tighten_compressed_output",
    "compress_history_impl",
    "compress_recent_window_impl",
    "compress_tool_results_impl",
    "inject_system_additions_impl",
    "tok_tool_result_impl",
    "truncate_large_result",
]

# Tests monkeypatch this module-level threshold directly.
TOOL_COMPRESS_THRESHOLD = 0


def _sync_threshold() -> None:
    """Synchronize the threshold between pipeline modules."""
    _history.TOOL_COMPRESS_THRESHOLD = TOOL_COMPRESS_THRESHOLD


def tok_tool_result_impl(
    content: str,
    compression_level: str = "balanced",
    tool_context: dict[str, Any] | None = None,
) -> str:
    """Compress a tool result using the history pipeline."""
    _sync_threshold()
    return _history.tok_tool_result_impl(
        content,
        compression_level=compression_level,
        tool_context=tool_context,
    )


def compress_tool_results_impl(
    messages: list[dict[str, Any]],
    result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None = None,
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    compression_level: str = "balanced",
    semantic_hash_cache: dict[str, str] | None = None,
    bypass_result_cache: bool = False,
    hot_summary_records: dict[str, Any] | None = None,
    session_files_read: set[str] | None = None,
    files_fully_delivered: dict[str, int] | None = None,
    first_exact_evidence_seen: set[str] | None = None,
    current_turn: int | None = None,
    keep_turns_window: int | None = None,
    preserve_exact_search_evidence: bool = False,
    recently_edited_files: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compress tool results in messages using the history pipeline."""
    _sync_threshold()
    return _history.compress_tool_results_impl(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level=compression_level,
        semantic_hash_cache=semantic_hash_cache,
        bypass_result_cache=bypass_result_cache,
        hot_summary_records=hot_summary_records,
        session_files_read=session_files_read,
        files_fully_delivered=files_fully_delivered,
        first_exact_evidence_seen=first_exact_evidence_seen,
        current_turn=current_turn,
        keep_turns_window=keep_turns_window,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
        recently_edited_files=recently_edited_files,
    )


def compress_recent_window_impl(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    threshold: int = RECENT_WINDOW_THRESHOLD,
    tool_compatible: bool = False,
    first_exact_evidence_seen: set[str] | None = None,
    preserve_exact_search_evidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply content-aware compression to tool_result blocks in the recent window."""
    _sync_threshold()
    return _compress_recent_window_impl(
        messages,
        tool_use_id_to_context=tool_use_id_to_context,
        threshold=threshold,
        tool_compatible=tool_compatible,
        first_exact_evidence_seen=first_exact_evidence_seen,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
    )
