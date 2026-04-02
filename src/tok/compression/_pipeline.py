from __future__ import annotations

"""Compression pipeline facade and backward-compatible entrypoints."""

from typing import Any

from . import _should_include_tok_state  # noqa: F401
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
from . import _history_pipeline as _history
from ._history_pipeline import (
    _compress_git_log_impl,
    _detect_tool_content_type_impl,
    compress_history_impl,
    compress_recent_window_impl,
    inject_system_additions_impl,
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
    _history.TOOL_COMPRESS_THRESHOLD = TOOL_COMPRESS_THRESHOLD


def tok_tool_result_impl(
    content: str, compression_level: str = "balanced"
) -> str:
    _sync_threshold()
    return _history.tok_tool_result_impl(
        content, compression_level=compression_level
    )


def compress_tool_results_impl(
    messages: list[dict[str, Any]],
    result_cache: dict[str, tuple[str, str, float]] | None = None,
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    compression_level: str = "balanced",
    semantic_hash_cache: dict[str, str] | None = None,
    bypass_result_cache: bool = False,
    hot_summary_records: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _sync_threshold()
    return _history.compress_tool_results_impl(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level=compression_level,
        semantic_hash_cache=semantic_hash_cache,
        bypass_result_cache=bypass_result_cache,
        hot_summary_records=hot_summary_records,
    )
