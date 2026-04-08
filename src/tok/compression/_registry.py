"""Compression registry helpers for tool-result content families."""

from __future__ import annotations

from collections.abc import Callable

Compressor = Callable[[str], str]


def build_default_registry(
    *,
    compress_pytest: Compressor,
    compress_grep: Compressor,
    compress_git_diff: Compressor,
    compress_ls: Compressor,
    compress_install: Compressor,
    compress_git_log: Compressor,
    compress_repetitive: Compressor,
    compress_file_read: Compressor,
    compress_search_results: Compressor,
    compress_stack_traces: Compressor,
    compress_grep_context: Compressor,
    compress_config_json: Compressor,
    compress_ps_output: Compressor,
    compress_env_output: Compressor,
) -> dict[str, Compressor]:
    """Build a registry mapping content types to compressor functions."""
    return {
        "pytest": compress_pytest,
        "grep": compress_grep,
        "git_diff": compress_git_diff,
        "ls": compress_ls,
        "install": compress_install,
        "git_log": compress_git_log,
        "repetitive": compress_repetitive,
        "file": compress_file_read,
        "search_results": compress_search_results,
        "stack_trace": compress_stack_traces,
        "ps_output": compress_ps_output,
        "env_output": compress_env_output,
        "grep_context": compress_grep_context,
        "config_json": compress_config_json,
        "json_skeleton": compress_config_json,
    }
