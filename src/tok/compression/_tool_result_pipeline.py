"""Tool-result compression pipeline helpers.

This module hosts the registry-based `tok_tool_result_impl` and related helpers.
It is intentionally kept independent from the higher-level history orchestration
to make large-file decomposition safer.
"""

from __future__ import annotations

from typing import Any

from ._registry import build_default_registry
from ._tool_result_codecs import (
    _compress_config_json,
    _compress_env_ps,
    _compress_file_read,
    _compress_git_diff,
    _compress_git_log,
    _compress_grep,
    _compress_grep_context,
    _compress_install,
    _compress_ls,
    _compress_pytest,
    _compress_repetitive,
    _compress_search_results,
    _compress_stack_traces,
    _detect_tool_content_type,
    _is_tok_cli_command,
    _tighten_compressed_output,
    truncate_large_result,
)


def detect_tool_content_type_impl(text: str) -> str:
    return _detect_tool_content_type(text)


def compress_git_log_impl(text: str) -> str:
    return _compress_git_log(text)


def _tool_command_hint(tool_context: dict[str, Any] | None) -> str:
    if not isinstance(tool_context, dict):
        return ""
    args = tool_context.get("args")
    if isinstance(args, dict):
        for key in ("command", "cmd"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ("command", "cmd"):
        value = tool_context.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def tok_tool_result_impl(
    content: str,
    *,
    tool_compress_threshold: int,
    compression_level: str = "balanced",
    tool_context: dict[str, Any] | None = None,
    session: Any | None = None,
) -> str:
    """Compress a tool result using registry-based compressors."""
    if len(content) <= tool_compress_threshold:
        return content
    if _is_tok_cli_command(_tool_command_hint(tool_context)):
        return content

    kind = detect_tool_content_type_impl(content)
    original_chars = len(content)
    registry = build_default_registry(
        compress_pytest=lambda text: _compress_pytest(text, command=_tool_command_hint(tool_context)),
        compress_grep=_compress_grep,
        compress_git_diff=_compress_git_diff,
        compress_ls=_compress_ls,
        compress_install=_compress_install,
        compress_git_log=compress_git_log_impl,
        compress_repetitive=lambda text: _compress_repetitive(text, command=_tool_command_hint(tool_context)),
        compress_file_read=lambda text: _compress_file_read(text, tool_context=tool_context, session=session),
        compress_search_results=_compress_search_results,
        compress_stack_traces=_compress_stack_traces,
        compress_grep_context=_compress_grep_context,
        compress_config_json=_compress_config_json,
        compress_ps_output=lambda text: _compress_env_ps(text, "ps_output"),
        compress_env_output=lambda text: _compress_env_ps(text, "env_output"),
    )
    compressor = registry.get(kind)
    compressed = compressor(content) if compressor else content

    compressed = _tighten_compressed_output(kind, compressed, compression_level)

    already_compressed = kind == "file" and ("|> [" in compressed or "lines]" in compressed)
    compressed = truncate_large_result(compressed, already_compressed=already_compressed)

    saved = original_chars - len(compressed)
    if saved <= 0:
        return content

    if not compressed.startswith(">>>"):
        compressed = (
            f">>> tok_compressed:tool_result|type:{kind}"
            f"|original_chars:{original_chars}|saved_chars:{saved}\n" + compressed
        )

    return compressed
