"""Tool-name classification constants shared across compression and runtime."""

from __future__ import annotations

from typing import Any

FILE_LIKE_TOOLS = frozenset(
    {
        "view",
        "view_file",
        "read",
        "read_file",
        "cat",
        "open_file",
        "get_file",
    }
)

EDIT_LIKE_TOOLS = frozenset(
    {
        "edit",
        "write",
        "edit_file",
        "write_file",
        "apply_patch",
        "str_replace_based_edit_tool",
    }
)

SEARCH_LIKE_TOOLS = frozenset(
    {
        "grep",
        "grep_search",
        "search",
        "rg",
        "find_by_name",
        "glob",
        "find",
        "code_search",
    }
)

LISTING_LIKE_TOOLS = frozenset({"list_dir", "ls"})

COMMAND_LIKE_TOOLS = frozenset(
    {
        "bash",
        "run_bash",
        "sh",
        "run_terminal",
        "computer",
        "run",
        "shell",
        "zsh",
        "bash_script",
        "execute_command",
        "cmd",
        "terminal",
        "exec",
    }
)

_PRECISION_READ_ARG_KEYS = ("offset", "limit", "start", "end")


def is_precision_read_context(context: dict[str, Any] | None) -> bool:
    """Return ``True`` when *context* is a bounded (precision) file read.

    A precision read is a file-like tool call carrying an explicit ``offset`` /
    ``limit`` / ``start`` / ``end`` window. Such a call is an explicit
    exact-evidence request from the agent: it asked for those specific lines, so
    the bridge must return them verbatim rather than skeletonizing or
    lossily truncating them. Returning a truncated (omitted-middle) precision
    read would also desynchronize delivery from the overlap-delta coverage
    tracker, leaving the omitted span permanently invisible.
    """
    if not context:
        return False
    if str(context.get("name", "")).lower() not in FILE_LIKE_TOOLS:
        return False
    args = context.get("args")
    if not isinstance(args, dict):
        return False
    return any(key in args for key in _PRECISION_READ_ARG_KEYS)
