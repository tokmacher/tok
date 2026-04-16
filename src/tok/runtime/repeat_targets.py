"""Helpers for detecting and summarizing repeated logical target reacquisition."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tok.compression import FILE_LIKE_TOOLS

SEARCH_LIKE_TOOLS = frozenset({"grep", "grep_search", "search", "rg", "find_by_name", "glob", "find", "code_search"})
COMMAND_LIKE_TOOLS = frozenset({"bash", "sh", "run_terminal", "computer"})
LISTING_LIKE_TOOLS = frozenset({"list_dir", "ls"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SCOPE_RE = re.compile(r"^\s*(class |def |async def )")
_SEARCH_RESULT_LINE_RE = re.compile(r"^(.*?:\d+(?::\d+)?):\s*(.*)$")
_SEARCH_RESULT_CONTEXT_LINE_RE = re.compile(r"^[^\s:][^:]*-\d+-(.*)$")
_ASSIGNMENT_RE = re.compile(r"(?<![=!<>])=(?!=)")
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
    "file_current",
    "file_history",
    "search",
    "listing",
    "file_metadata",
    "unknown",
]

SearchResultEvidenceLevel = Literal["navigation", "exact_content"]
_EVIDENCE_VARIANT = Literal["full", "snippet", "diff", "metadata", "copy", "search_results"]
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
_TMP_PREFIXES = (tempfile.gettempdir(), "/tmp/", "/var/tmp/", "/private/tmp/", "tmp/")


@dataclass(frozen=True)
class EvidenceIntent:
    domain: _EVIDENCE_DOMAIN
    anchor: str
    variant: _EVIDENCE_VARIANT
    novelty_key: str
    display_label: str
    source_kind: _EVIDENCE_SOURCE_KIND


def _classify_revision(revision: str) -> str:
    """Classify a git revision string into a normalized form."""
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
    """Extract the file path and revision from a git history command."""
    text = " ".join(str(command or "").strip().split())
    if not text:
        return None, ""
    m = _GIT_SHOW_RE.match(text)
    if m:
        return normalize_path_target(m.group(2)), _classify_revision(m.group(1))
    m = _GIT_DIFF_PATH_RE.match(text)
    if m:
        path_str = m.group(2).strip()
        return normalize_path_target(path_str), _classify_revision(m.group(1))
    return None, ""


def extract_shell_search_params(
    command: str,
) -> tuple[str | None, str | None]:
    """Extract search query and scope from a shell search command."""
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
    """Extract the metadata subtype from a metadata probe command."""
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
    """Check if a path is within a temporary directory."""
    normalized = normalize_path_target(path)
    return any(normalized.startswith(p) for p in _TMP_PREFIXES)


def _resolve_command_intent(command: str, path: str | None) -> EvidenceIntent | None:
    """Resolve evidence intent for command-like tools."""
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
            display_label=(f"{shell_query} @ {scope}" if scope else shell_query),
            source_kind="shell_search",
        )

    meta_subtype = extract_metadata_probe(command)
    if meta_subtype:
        meta_path = path or ""
        domain: _EVIDENCE_DOMAIN = "listing" if meta_subtype in {"ls", "find"} else "file_metadata"
        return EvidenceIntent(
            domain=domain,
            anchor=normalize_path_target(meta_path) or meta_subtype,
            variant="metadata",
            novelty_key=meta_subtype,
            display_label=(f"{meta_subtype}:{meta_path}" if meta_path else meta_subtype) or "",
            source_kind="metadata_probe",
        )

    shell_read_path = extract_shell_file_read_path(command)
    if shell_read_path:
        is_temp = _is_temp_path(shell_read_path)
        return EvidenceIntent(
            domain="file_current",
            anchor=(normalize_path_target(shell_read_path) or "path-missing"),
            variant="copy" if is_temp else "full",
            novelty_key="",
            display_label=shell_read_path,
            source_kind="temp_copy" if is_temp else "shell_read",
        )
    return None


def _resolve_native_intent(
    tool_name: str,
    path: str | None,
    query: str | None,
) -> EvidenceIntent | None:
    """Resolve evidence intent for native tools (file, listing, search)."""
    lowered = tool_name

    if lowered in FILE_LIKE_TOOLS and path:
        return EvidenceIntent(
            domain="file_current",
            anchor=normalize_path_target(path) or "path-missing",
            variant="full",
            novelty_key="",
            display_label=path,
            source_kind="native_tool",
        )

    if lowered in LISTING_LIKE_TOOLS and path:
        return EvidenceIntent(
            domain="listing",
            anchor=normalize_path_target(path) or "path-missing",
            variant="search_results",
            novelty_key="",
            display_label=path,
            source_kind="native_tool",
        )

    if lowered in SEARCH_LIKE_TOOLS or query:
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
            display_label=f"{query or ''} @ {scope}" if scope else query or "",
            source_kind="native_tool",
        )
    return None


def resolve_evidence_intent(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
) -> EvidenceIntent | None:
    """Resolve the evidence intent for a tool invocation."""
    lowered = str(tool_name or "").strip().lower()

    if lowered in COMMAND_LIKE_TOOLS and command:
        intent = _resolve_command_intent(command, path)
        if intent:
            return intent

    return _resolve_native_intent(lowered, path, query)


_EVIDENCE_KEYS = frozenset({"line", "snippet", "content", "text", "match", "context"})


def _has_line_evidence(lines: list[str]) -> bool:
    """Check if any line matches search result evidence patterns."""
    return any(_SEARCH_RESULT_LINE_RE.match(line) for line in lines) or any(
        _SEARCH_RESULT_CONTEXT_LINE_RE.match(line) for line in lines
    )


def _has_json_array_evidence(data: list[Any]) -> bool:
    """Check if JSON array contains items with evidence keys."""
    for item in data:
        if not isinstance(item, dict):
            continue
        keys = {str(key).lower() for key in item}
        if _EVIDENCE_KEYS.intersection(keys):
            return True
    return False


def search_result_evidence_level(text: str) -> SearchResultEvidenceLevel:
    """Classify search output as navigational or exact content evidence."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return "navigation"

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return "navigation"

    if _has_line_evidence(lines):
        return "exact_content"

    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            data = json.loads(cleaned)
        except Exception:
            return "navigation"
        if isinstance(data, list) and _has_json_array_evidence(data):
            return "exact_content"
        return "navigation"

    return "navigation"


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
    """Extract file path from cat command arguments."""
    if args[:1] == ["--"]:
        args = args[1:]
    if len(args) != 1:
        return None
    return args[0]


def _extract_shell_read_head_tail_nl_path(args: list[str]) -> str | None:
    """Extract file path from head/tail/nl command arguments."""
    if not args:
        return None
    candidate_paths = [arg for arg in args if arg != "--" and not arg.startswith("-")]
    if len(candidate_paths) != 1:
        return None
    return candidate_paths[0]


def _extract_shell_read_sed_path(args: list[str]) -> str | None:
    """Extract file path from sed command arguments."""
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
    """Normalize a file path to POSIX format."""
    text = str(path or "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    if text.startswith("/"):
        return posixpath.normpath(text)
    normalized = posixpath.normpath(text)
    return "" if normalized == "." else normalized


def normalize_command_family(command: str) -> str:
    """Normalize a command to its canonical family name."""
    text = " ".join(str(command or "").strip().split())
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    while parts and Path(parts[0]).name.lower() in {"env", "/usr/bin/env"}:
        parts = parts[1:]
    if len(parts) >= 2 and Path(parts[0]).name.lower() == "uv" and parts[1] == "run":
        parts = parts[2:]
    if (
        len(parts) >= 3
        and Path(parts[0]).name.lower()
        in {
            "python",
            "python3",
        }
        and parts[1] == "-m"
    ):
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
    """Normalize a tool to its canonical family name."""
    lowered = str(tool_name or "").strip().lower()
    if lowered in FILE_LIKE_TOOLS:
        return "file_read"
    if lowered in LISTING_LIKE_TOOLS:
        return "listing"
    if lowered in COMMAND_LIKE_TOOLS and extract_shell_file_read_path(command or ""):
        return "file_read"
    if lowered in SEARCH_LIKE_TOOLS or query:
        return "search"
    if lowered in COMMAND_LIKE_TOOLS or command:
        return "command"
    return lowered or "unknown"


def _normalize_identity_args(
    args: dict[str, Any] | None,
    *,
    excluded_keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if key in excluded_keys:
            continue
        if isinstance(value, str):
            cleaned = " ".join(value.split())
            if cleaned:
                normalized[key] = cleaned
            continue
        if isinstance(value, int | float | bool):
            normalized[key] = value
            continue
        if value is not None:
            normalized[key] = str(value).strip()
    return normalized


def _normalize_command_parts(parts: list[str]) -> list[str]:
    """Normalize command parts by stripping env wrappers and uv run."""
    while parts and Path(parts[0]).name.lower() in {"env", "/usr/bin/env"}:
        parts = parts[1:]
    if len(parts) >= 2 and Path(parts[0]).name.lower() == "uv" and parts[1] == "run":
        parts = parts[2:]
    return parts


def _parse_listing_args(parts: list[str]) -> tuple[str, str]:
    """
    Parse listing command arguments to extract path and mode.

    Returns (listing_path, mode_string).
    """
    listing_path = ""
    mode_parts: list[str] = []
    for arg in parts:
        if arg == "--":
            continue
        if arg.startswith("-"):
            mode_parts.append(arg)
            continue
        if not listing_path:
            listing_path = arg
        else:
            mode_parts.append(arg)
    return listing_path, " ".join(mode_parts).strip()


def _extract_shell_listing_target(command: str) -> tuple[str, str]:
    """Extract the listing target from a shell command."""
    text = " ".join(str(command or "").strip().split())
    if not text:
        return "", ""
    try:
        parts = shlex.split(text)
    except Exception:
        return "", ""
    parts = _normalize_command_parts(parts)
    if not parts:
        return "", ""
    command_name = Path(parts[0]).name.lower()
    if command_name not in {"ls", "find"}:
        return "", ""
    return _parse_listing_args(parts[1:])


def evidence_identity_key(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
    args: dict[str, Any] | None = None,
) -> str | None:
    """Return the exact-observation identity key for an evidence target."""
    lowered = str(tool_name or "").strip().lower()
    tool_args = args if isinstance(args, dict) else {}
    excluded = frozenset(
        {
            "path",
            "file_path",
            "AbsolutePath",
            "TargetFile",
            "SearchDirectory",  # find_by_name
            "search_folder_absolute_uri",  # code_search
            "query",
            "pattern",
            "Pattern",  # find_by_name
            "search",
            "search_term",  # code_search
            "text",
            "command",
            "cmd",
            "offset",
            "limit",
            "start",
            "end",
        }
    )

    shell_file_read_path = extract_shell_file_read_path(command or "")
    if lowered in FILE_LIKE_TOOLS or shell_file_read_path:
        resolved_path = path or shell_file_read_path or ""
        return "file_read|" + (normalize_path_target(resolved_path) or "path-missing")

    shell_query, shell_scope = extract_shell_search_params(command or "")
    if lowered in SEARCH_LIKE_TOOLS or shell_query:
        query_value = query or shell_query or ""
        scope_value = normalize_path_target(path or shell_scope or "")
        payload = {
            "flags": _normalize_identity_args(
                tool_args,
                excluded_keys=excluded,
            ),
            "query": " ".join(str(query_value or "").split()),
            "scope": scope_value,
        }
        return "search|" + json.dumps(payload, sort_keys=True, separators=(",", ":"))

    shell_listing_path, shell_listing_mode = _extract_shell_listing_target(command or "")
    if lowered in LISTING_LIKE_TOOLS or shell_listing_path:
        listing_path = normalize_path_target(path or shell_listing_path or "")
        mode_payload = (
            {"command_mode": shell_listing_mode}
            if shell_listing_mode
            else _normalize_identity_args(
                tool_args,
                excluded_keys=excluded,
            )
        )
        payload = {
            "mode": mode_payload,
            "path": listing_path or "path-missing",
        }
        return "listing|" + json.dumps(payload, sort_keys=True, separators=(",", ":"))

    return None


def logical_target_identity(
    tool_name: str,
    *,
    path: str | None = None,
    query: str | None = None,
    command: str | None = None,
    args: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Generate a canonical logical target identity for a tool invocation."""
    family = normalize_tool_family(tool_name, query=query, command=command)
    if family == "file_read":
        resolved_path = path or extract_shell_file_read_path(command or "") or ""
        return family, normalize_path_target(resolved_path) or "path-missing"
    if family == "search":
        payload = {
            "query": " ".join(str(query or "").split()),
            "search_path": normalize_path_target(path or ""),
        }
        return family, json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if family == "listing":
        shell_path, shell_mode = _extract_shell_listing_target(command or "")
        target_path = normalize_path_target(path or shell_path or "") or "path-missing"
        normalized_args = _normalize_identity_args(
            args if isinstance(args, dict) else {},
            excluded_keys=frozenset(
                {
                    "path",
                    "file_path",
                    "AbsolutePath",
                    "TargetFile",
                    "text",
                    "command",
                    "cmd",
                }
            ),
        )
        payload = {
            "mode": shell_mode or json.dumps({k: str(v) for k, v in normalized_args.items()}),
            "path": target_path,
        }
        return family, json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if family == "command":
        payload = {"family": normalize_command_family(command or "")}
        return family, json.dumps(payload, sort_keys=True, separators=(",", ":"))
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
    """Generate a human-readable display label for a target."""
    if family == "file_read":
        resolved_path = path or extract_shell_file_read_path(command or "") or ""
        return str(resolved_path or logical_target or "").strip()
    if family == "search":
        q = " ".join(str(query or "").split())
        p = str(path or "").strip()
        if q and p:
            return f"{q} @ {p}"
        return q or logical_target
    if family == "listing":
        p = str(path or "").strip()
        return p or logical_target
    if family == "command":
        family_name = normalize_command_family(command or "")
        return family_name or logical_target
    return logical_target


def stable_digest(text: str) -> str:
    """Generate a stable 16-character digest of text."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def _truncate_chars(text: str, limit: int) -> str:
    """Truncate text to a character limit with ellipsis."""
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 1:
        return cleaned[:limit]
    return cleaned[: limit - 1].rstrip() + "…"


def _join_summary_lines(lines: list[str], limit: int) -> str:
    """Join summary lines with pipe separators within a character limit."""
    parts: list[str] = []
    for line in lines:
        item = line.strip()
        if not item:
            continue
        candidate = " | ".join([*parts, item])
        if len(candidate) > limit:
            break
        parts.append(item)
    return _truncate_chars(" | ".join(parts), limit)


def _non_empty_lines(text: str) -> list[str]:
    """Extract non-empty lines from text."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _summary_scoring_text(line: str) -> str:
    stripped = str(line or "").strip()
    if not stripped:
        return ""
    match = _SEARCH_RESULT_LINE_RE.match(stripped)
    if match:
        payload = match.group(2).strip()
        if payload:
            return payload
    return stripped


# Scoring lookup tables for _summary_line_score
_HIGH_SCORE_PREFIXES = [
    (("def ", "async def ", "class "), 120),
    (("return ", "raise ", "yield "), 110),
    (("import ", "from "), 90),
    (
        (
            "if ",
            "elif ",
            "else:",
            "for ",
            "while ",
            "try:",
            "except ",
            "with ",
            "match ",
            "case ",
        ),
        70,
    ),
]
_LOW_SCORE_PREFIXES = [("@", 20)]
_NEGATIVE_SCORE_PREFIXES = [
    (("pass", "continue", "break"), 80),
    ("#", 40),
]
_NEGATIVE_TOKENS = frozenset(("logger.debug", "logger.info", "logger.warning", "logger.error", "print("))
_NEGATIVE_EXCEPTIONS = frozenset(("except Exception", "except BaseException"))


def _calculate_positive_score(_text: str, lower: str) -> int:
    """Calculate positive score contribution from prefixes."""
    score = 0
    for prefixes, points in _HIGH_SCORE_PREFIXES:
        if lower.startswith(prefixes):
            score += points
    for prefix, points in _LOW_SCORE_PREFIXES:
        if lower.startswith(prefix):
            score += points
    return score


def _calculate_negative_score(text: str, lower: str) -> int:
    """Calculate negative score contribution."""
    score = 0
    for prefixes, points in _NEGATIVE_SCORE_PREFIXES:
        if isinstance(prefixes, (tuple, str)):
            if lower.startswith(prefixes):
                score -= points
    for exc in _NEGATIVE_EXCEPTIONS:
        if exc in text:
            score -= 80
    if any(token in text for token in _NEGATIVE_TOKENS):
        score -= 60
    return score


def _summary_line_score(line: str) -> int:
    text = _summary_scoring_text(line)
    if not text:
        return -1000

    score = 0
    lower = text.lstrip()

    score += _calculate_positive_score(text, lower)

    if _ASSIGNMENT_RE.search(text) and not any(token in text for token in ("==", "!=", ">=", "<=")):
        score += 100
    if "(" in text and ")" in text and not lower.startswith("#"):
        score += 30

    score += _calculate_negative_score(text, lower)

    return score


def _structural_summary_indices(lines: list[str], max_lines: int) -> list[int]:
    scored: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        score = _summary_line_score(line)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return []

    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_lines]
    return sorted({idx for _, idx in top})


def _find_enclosing_scope(all_lines: list[str], summary_lines: list[str]) -> str:
    if not summary_lines or not all_lines:
        return ""
    first_line = summary_lines[0].strip()
    for i, line in enumerate(all_lines):
        if line.strip() == first_line:
            for j in range(i - 1, max(i - 30, -1), -1):
                if _SCOPE_RE.match(all_lines[j]):
                    return all_lines[j].strip().rstrip(":")
            break
    return ""


def build_file_summary(text: str, *, max_chars: int, max_lines: int) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""
    structural_indices = _structural_summary_indices(lines, max_lines)
    all_lines = str(text or "").splitlines()
    if structural_indices:
        head = [lines[idx] for idx in structural_indices]
        scope_prefix = _find_enclosing_scope(all_lines, head)
        if scope_prefix and scope_prefix not in head:
            head.insert(0, scope_prefix)
        return _join_summary_lines(head, max_chars)

    head = lines[: min(8, max_lines)]
    tail_count = min(4, max(0, max_lines - len(head)))
    if len(lines) > 12 and tail_count:
        head.extend(lines[-tail_count:])
    scope_prefix = _find_enclosing_scope(all_lines, head)
    if scope_prefix and scope_prefix not in head:
        head.insert(0, scope_prefix)
    return _join_summary_lines(head, max_chars)


def build_file_skeleton(text: str, *, max_chars: int, max_lines: int) -> str:
    """Return a deterministic scope signature list for a file."""
    all_lines = str(text or "").splitlines()
    if not all_lines:
        return ""
    signatures: list[str] = []
    seen: set[str] = set()
    for line in all_lines:
        if not _SCOPE_RE.match(line):
            continue
        cleaned = line.strip().rstrip(":")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        signatures.append(cleaned)
        if len(signatures) >= max_lines:
            break
    return _join_summary_lines(signatures, max_chars)


def build_search_summary(text: str, *, max_chars: int, max_lines: int) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""

    structural_indices = _structural_summary_indices(lines, max_lines)
    if structural_indices:
        selected = [lines[idx] for idx in structural_indices]
        return _join_summary_lines(selected, max_chars)

    return _join_summary_lines(lines[:max_lines], max_chars)


def canonicalize_command_output(text: str) -> str:
    """Canonicalize command output by normalizing newlines and ANSI codes."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _ANSI_RE.sub("", cleaned)
    stripped = cleaned.strip()
    if stripped and stripped[0] in "[{":
        try:
            return json.dumps(json.loads(stripped), sort_keys=True, separators=(",", ":"))
        except Exception:
            pass
    return cleaned


def build_command_summary(text: str, *, max_chars: int, max_lines: int) -> str:
    """Build a summary of command output."""
    canonical = canonicalize_command_output(text)
    return _join_summary_lines(_non_empty_lines(canonical)[:max_lines], max_chars)


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
    """Build a summary of tool output based on its family type."""
    if family == "file_read":
        return build_file_summary(text, max_chars=file_max_chars, max_lines=file_max_lines)
    if family == "search":
        return build_search_summary(text, max_chars=search_max_chars, max_lines=search_max_lines)
    if family == "listing":
        return build_search_summary(text, max_chars=search_max_chars, max_lines=search_max_lines)
    if family == "command":
        return build_command_summary(text, max_chars=command_max_chars, max_lines=command_max_lines)
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
    exact_evidence_key: str = ""
    hot_promotion_turn: int = 0
    stuck_promotion_turn: int = 0
    last_injected_turn: int = 0
    repeat_count: int = 0
    recent_window_count: int = 0
    stuck_window_count: int = 0
    unchanged_result_count: int = 0
    evidence_intent: EvidenceIntent | None = None
