"""Tool-name classification constants shared across compression and runtime."""

from __future__ import annotations

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
