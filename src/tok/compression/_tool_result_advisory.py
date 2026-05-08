"""Advisories for expensive search-like tool outputs.

These footers are informational only: they do not affect evidence policy or
compression behavior. They exist to nudge callers toward scoped searches when
results are excessively large.
"""

from __future__ import annotations

import re
import threading

_TOK_CLI_TOKEN_RE = re.compile(r"(?<![\w./-])tok(?![\w-])")

# Thresholds for search-cost advisory
_GREP_ADVISORY_MATCH_THRESHOLD = 50
_GREP_ADVISORY_FILE_THRESHOLD = 10
_GREP_ADVISORY_TOKEN_THRESHOLD = 2000  # Estimated tokens

# Advisory cooldown state (per-query identity -> last advisory turn)
_advisory_cooldown: dict[str, int] = {}
_advisory_lock = threading.Lock()
_ADVISORY_COOLDOWN_TURNS = 3


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token on average for code."""
    return len(text) // 4


def _is_tok_cli_command(command: str) -> bool:
    """
    Best-effort detection of tok CLI invocations in shell command strings.

    Handles direct calls (`tok stats`) and common wrappers (`env X=1 tok ...`,
    `bash -lc 'tok ...'`, `uv run tok ...`).
    """
    cleaned = " ".join((command or "").strip().split())
    if not cleaned:
        return False
    if cleaned.startswith("tok "):
        return True
    return bool(_TOK_CLI_TOKEN_RE.search(cleaned))


def _build_search_advisory(
    match_count: int,
    file_count: int,
    estimated_tokens: int = 0,
    has_scope: bool = True,
    query_identity: str | None = None,
    current_turn: int = 0,
) -> str:
    """
    Build advisory footer for expensive search results.

    Returns an advisory footer string, or empty string if no advisory warranted.
    """
    global _advisory_cooldown

    with _advisory_lock:
        if query_identity and query_identity in _advisory_cooldown:
            last_turn = _advisory_cooldown[query_identity]
            if current_turn - last_turn < _ADVISORY_COOLDOWN_TURNS:
                return ""  # Still in cooldown

    # Determine if advisory is warranted
    high_matches = match_count > _GREP_ADVISORY_MATCH_THRESHOLD
    many_files = file_count > _GREP_ADVISORY_FILE_THRESHOLD
    high_tokens = estimated_tokens > _GREP_ADVISORY_TOKEN_THRESHOLD
    unscoped = not has_scope

    if not (high_matches or many_files or high_tokens or unscoped):
        return ""

    hints: list[str] = []

    if unscoped and (high_matches or many_files):
        hints.append("unscoped search")
        if many_files:
            hints.append("try path: or glob: filter")
        else:
            hints.append("narrow with path or pattern")
    elif many_files and high_matches:
        hints.append(f"{file_count} files")
        hints.append("try path: or glob: filter")
    elif high_matches:
        hints.append(f"{match_count} matches")
        hints.append("consider narrower pattern")
    elif many_files:
        hints.append(f"{file_count} files")
        hints.append("try path: filter")
    elif high_tokens:
        hints.append("large result")
        hints.append("consider narrowing scope")
    else:
        hints.append("broad search")
        hints.append("consider narrowing scope")

    advisory = f"[tok advisory: {' - '.join(hints)}]"

    # Record in cooldown
    if query_identity:
        with _advisory_lock:
            _advisory_cooldown[query_identity] = current_turn

    return advisory


def clear_advisory_cooldown() -> None:
    """Clear the advisory cooldown cache. Call between sessions."""
    global _advisory_cooldown
    with _advisory_lock:
        _advisory_cooldown = {}
