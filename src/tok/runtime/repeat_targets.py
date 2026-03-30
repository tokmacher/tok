"""Helpers for detecting and summarizing repeated logical target reacquisition."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..compression import FILE_LIKE_TOOLS

SEARCH_LIKE_TOOLS = frozenset({"grep", "grep_search", "search", "rg"})
COMMAND_LIKE_TOOLS = frozenset({"bash", "sh", "run_terminal", "computer"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SHELL_UNSAFE_MARKERS = (
    "&&",
    "||",
    ";",
    "|",
    ">>",
    ">",
    "<",
    "$(",
    "`",
    "*",
    "?",
)

_EVIDENCE_DOMAIN = Literal[
    "file_current", "file_history", "search", "file_metadata", "unknown"
]
_EVIDENCE_VARIANT = Literal[
    "full", "snippet", "diff", "metadata", "copy", "search_results"
]
_EVIDENCE_SOURCE_KIND = Literal[
    "native_tool",
    "shell_read",
    "shell_search",
    "git_history",
    "temp_copy",
    "metadata_probe",
]

_GIT_SHOW_RE = re.compile(r"^git\s+show\s+([^:]+):(.+)$")
_GIT_DIFF_PATH_RE = re.compile(r"^git\s+diff\s+([\w@~^./:-]+)\s*--\s*(.+)$")
_METADATA_SUBTYPES = frozenset({"git", "stat", "wc", "file", "ls", "find"})
_TMP_PREFIXES = ("/tmp/", "/var/tmp/", "/private/tmp/", "tmp/")


_TMP_PREFIXES = ("/tmp/", "/var/tmp/", "/private/tmp/", "tmp/")


@dataclass(frozen=True)
class EvidenceIntent:
    domain: _EVIDENCE_DOMAIN
    anchor: str
    variant: _EVIDENCE_VARIANT
    novelty_key: str
    display_label: str
    source_kind: _EVIDENCE_SOURCE_KIND


def _classify_revision(revision: str) -> str:
    if revision in {"HEAD", "head"}:
        return "HEAD"
    if revision.startswith("HEAD~"):
        return "HEAD~N"
    if re.match(r"^[0-9a-f]{4,40}$", revision):
        return "sha"
    if revision.startswith("@"):
        return "ref"
    return "other"


def extract_git_history_path(command: str) -> tuple[str | None, str]:
    text = " ".join(str(command or "").strip().split())
    if not text:
        return None, ""
    m = _GIT_SHOW_RE.match(text)
    if m:
        return normalize_path_target(m.group(2)), _classify_revision(
            m.group(1)
        )
    m = _GIT_DIFF_PATH_RE.match(text)
    if m:
        path_str = m.group(2).strip()
        return normalize_path_target(path_str), _classify_revision(m.group(1))
    return None, ""


def extract_shell_search_params(
    command: str,
) -> tuple[str | None, str | None]:
    text = " ".join(str(command or "").strip().split())
    if not text:
        return None, None
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    if not parts:
        return None, None
    cmd = Path(parts[0]).name.lower()
    if cmd not in {"grep", "rg", "ag"}:
        return None, None
    args = [a for a in parts[1:] if not a.startswith("-") and a != "--"]
    if not args:
        return None, None
    if len(args) == 1:
        return args[0], None
    return args[0], normalize_path_target(args[-1])


def extract_metadata_probe(command: str) -> str | None:
    text = " ".join(str(command or "").strip().split())
    if not text:
        return None
    if any(marker in text for marker in _SHELL_UNSAFE_MARKERS):
        return None
    try:
        parts = shlex.split(text)
    except Exception:
        return None
    if not parts:
        return None
    cmd = Path(parts[0]).name.lower()
    if cmd == "git" and len(parts) >= 2:
        sub = parts[1].lower()
        if sub in {"log", "status", "shortlog", "blame"}:
            return f"git_{sub}"
        return None
    if cmd in _METADATA_SUBTYPES:
        return cmd
    return None


def _is_temp_path(path: str) -> bool:
    normalized = normalize_path_target(path)
    return any(normalized.startswith(p) for p in _TMP_PREFIXES)


def resolve_evidence_intent(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
) -> EvidenceIntent | None:
    lowered = str(tool_name or "").strip().lower()

    if lowered in COMMAND_LIKE_TOOLS and command:
        git_path, git_rev = extract_git_history_path(command)
        if git_path:
            return EvidenceIntent(
                domain="file_history",
                anchor=git_path,
                variant="diff",
                novelty_key=git_rev,
                display_label=f"{git_path}@{git_rev}",
                source_kind="git_history",
            )

        shell_query, shell_scope = extract_shell_search_params(command)
        if shell_query:
            scope = shell_scope or ""
            anchor = json.dumps(
                {"query": shell_query, "scope": scope},
                sort_keys=True,
                separators=(",", ":"),
            )
            return EvidenceIntent(
                domain="search",
                anchor=anchor,
                variant="search_results",
                novelty_key=anchor,
                display_label=(
                    f"{shell_query} @ {scope}" if scope else shell_query
                ),
                source_kind="shell_search",
            )

        meta_subtype = extract_metadata_probe(command)
        if meta_subtype:
            meta_path = path or ""
            return EvidenceIntent(
                domain="file_metadata",
                anchor=normalize_path_target(meta_path) or meta_subtype,
                variant="metadata",
                novelty_key=meta_subtype,
                display_label=(
                    f"{meta_subtype}:{meta_path}"
                    if meta_path
                    else meta_subtype
                ),
                source_kind="metadata_probe",
            )

        shell_read_path = extract_shell_file_read_path(command)
        if shell_read_path:
            is_temp = _is_temp_path(shell_read_path)
            return EvidenceIntent(
                domain="file_current",
                anchor=(
                    normalize_path_target(shell_read_path) or "path-missing"
                ),
                variant="copy" if is_temp else "full",
                novelty_key="",
                display_label=shell_read_path,
                source_kind="temp_copy" if is_temp else "shell_read",
            )

    if lowered in FILE_LIKE_TOOLS and path:
        return EvidenceIntent(
            domain="file_current",
            anchor=normalize_path_target(path) or "path-missing",
            variant="full",
            novelty_key="",
            display_label=path,
            source_kind="native_tool",
        )

    if lowered in SEARCH_LIKE_TOOLS and query:
        scope = normalize_path_target(path or "") or ""
        anchor = json.dumps(
            {"query": query, "scope": scope},
            sort_keys=True,
            separators=(",", ":"),
        )
        return EvidenceIntent(
            domain="search",
            anchor=anchor,
            variant="search_results",
            novelty_key=anchor,
            display_label=f"{query} @ {scope}" if scope else query,
            source_kind="native_tool",
        )

    return None


def extract_shell_file_read_path(command: str) -> str | None:
    """Return the direct read-only file target for simple shell inspection commands."""
    text = " ".join(str(command or "").strip().split())
    if not text:
        return None
    if any(marker in text for marker in _SHELL_UNSAFE_MARKERS):
        return None
    try:
        parts = shlex.split(text)
    except Exception:
        return None
    while parts and Path(parts[0]).name.lower() in {"env", "/usr/bin/env"}:
        parts = parts[1:]
    if not parts:
        return None
    command_name = Path(parts[0]).name.lower()
    args = parts[1:]
    handler = _SHELL_READ_PATH_HANDLERS.get(command_name)
    if not handler:
        return None
    return handler(args)


def _extract_shell_read_cat_path(args: list[str]) -> str | None:
    if args[:1] == ["--"]:
        args = args[1:]
    if len(args) != 1:
        return None
    return args[0]


def _extract_shell_read_head_tail_nl_path(args: list[str]) -> str | None:
    if not args:
        return None
    candidate_paths = [
        arg for arg in args if arg != "--" and not arg.startswith("-")
    ]
    if len(candidate_paths) != 1:
        return None
    return candidate_paths[0]


def _extract_shell_read_sed_path(args: list[str]) -> str | None:
    if len(args) != 3 or args[0] != "-n":
        return None
    return args[2]


_SHELL_READ_PATH_HANDLERS = {
    "cat": _extract_shell_read_cat_path,
    "head": _extract_shell_read_head_tail_nl_path,
    "tail": _extract_shell_read_head_tail_nl_path,
    "nl": _extract_shell_read_head_tail_nl_path,
    "sed": _extract_shell_read_sed_path,
}


def normalize_path_target(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    if text.startswith("/"):
        return posixpath.normpath(text)
    normalized = posixpath.normpath(text)
    return "" if normalized == "." else normalized


def normalize_command_family(command: str) -> str:
    text = " ".join(str(command or "").strip().split())
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    while parts and Path(parts[0]).name.lower() in {"env", "/usr/bin/env"}:
        parts = parts[1:]
    if (
        len(parts) >= 2
        and Path(parts[0]).name.lower() == "uv"
        and parts[1] == "run"
    ):
        parts = parts[2:]
    if len(parts) >= 3 and Path(parts[0]).name.lower() in {
        "python",
        "python3",
    }:
        if parts[1] == "-m":
            return parts[2].lower()
    if not parts:
        return text.lower()
    return Path(parts[0]).name.lower()


def normalize_tool_family(
    tool_name: str,
    *,
    query: str | None = None,
    command: str | None = None,
) -> str:
    lowered = str(tool_name or "").strip().lower()
    if lowered in FILE_LIKE_TOOLS:
        return "file_read"
    if lowered in COMMAND_LIKE_TOOLS and extract_shell_file_read_path(
        command or ""
    ):
        return "file_read"
    if lowered in SEARCH_LIKE_TOOLS or query:
        return "search"
    if lowered in COMMAND_LIKE_TOOLS or command:
        return "command"
    return lowered or "unknown"


def logical_target_identity(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
) -> tuple[str, str]:
    family = normalize_tool_family(tool_name, query=query, command=command)
    if family == "file_read":
        resolved_path = (
            path or extract_shell_file_read_path(command or "") or ""
        )
        return family, normalize_path_target(resolved_path) or "path-missing"
    if family == "search":
        payload = {
            "query": " ".join(str(query or "").split()),
            "search_path": normalize_path_target(path or ""),
        }
        return family, json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
    if family == "command":
        payload = {"family": normalize_command_family(command or "")}
        return family, json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
    raw = json.dumps(
        {
            "path": normalize_path_target(path or ""),
            "query": " ".join(str(query or "").split()),
            "command": " ".join(str(command or "").split()),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return family, raw


def display_target_label(
    family: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
    logical_target: str = "",
) -> str:
    if family == "file_read":
        resolved_path = (
            path or extract_shell_file_read_path(command or "") or ""
        )
        return str(resolved_path or logical_target or "").strip()
    if family == "search":
        q = " ".join(str(query or "").split())
        p = str(path or "").strip()
        if q and p:
            return f"{q} @ {p}"
        return q or logical_target
    if family == "command":
        family_name = normalize_command_family(command or "")
        return family_name or logical_target
    return logical_target


def stable_digest(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def _truncate_chars(text: str, limit: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 1:
        return cleaned[:limit]
    return cleaned[: limit - 1].rstrip() + "…"


def _join_summary_lines(lines: list[str], limit: int) -> str:
    parts: list[str] = []
    for line in lines:
        item = line.strip()
        if not item:
            continue
        candidate = " | ".join(parts + [item])
        if len(candidate) > limit:
            break
        parts.append(item)
    return _truncate_chars(" | ".join(parts), limit)


def _non_empty_lines(text: str) -> list[str]:
    return [
        line.strip() for line in str(text or "").splitlines() if line.strip()
    ]


def build_file_summary(text: str, *, max_chars: int, max_lines: int) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""
    head = lines[: min(8, max_lines)]
    tail_count = min(4, max(0, max_lines - len(head)))
    if len(lines) > 12 and tail_count:
        head.extend(lines[-tail_count:])
    return _join_summary_lines(head, max_chars)


def build_search_summary(text: str, *, max_chars: int, max_lines: int) -> str:
    return _join_summary_lines(_non_empty_lines(text)[:max_lines], max_chars)


def canonicalize_command_output(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _ANSI_RE.sub("", cleaned)
    stripped = cleaned.strip()
    if stripped and stripped[0] in "[{":
        try:
            return json.dumps(
                json.loads(stripped), sort_keys=True, separators=(",", ":")
            )
        except Exception:
            pass
    return cleaned


def build_command_summary(text: str, *, max_chars: int, max_lines: int) -> str:
    canonical = canonicalize_command_output(text)
    return _join_summary_lines(
        _non_empty_lines(canonical)[:max_lines], max_chars
    )


def build_summary_for_family(
    family: str,
    text: str,
    *,
    file_max_chars: int,
    file_max_lines: int,
    search_max_chars: int,
    search_max_lines: int,
    command_max_chars: int,
    command_max_lines: int,
) -> str:
    if family == "file_read":
        return build_file_summary(
            text, max_chars=file_max_chars, max_lines=file_max_lines
        )
    if family == "search":
        return build_search_summary(
            text, max_chars=search_max_chars, max_lines=search_max_lines
        )
    if family == "command":
        return build_command_summary(
            text, max_chars=command_max_chars, max_lines=command_max_lines
        )
    return ""


@dataclass
class RepeatTargetEvent:
    turn_index: int
    tool_family: str
    logical_target: str
    display_target: str
    token_cost: int
    result_digest: str
    unchanged_result: bool = False
    evidence_anchor: str = ""


@dataclass
class HotSummaryRecord:
    tool_family: str
    logical_target: str
    display_target: str
    summary: str
    token_cost: int
    result_digest: str
    last_seen_turn: int
    hot_promotion_turn: int = 0
    stuck_promotion_turn: int = 0
    last_injected_turn: int = 0
    repeat_count: int = 0
    recent_window_count: int = 0
    stuck_window_count: int = 0
    unchanged_result_count: int = 0
    evidence_intent: EvidenceIntent | None = None
