"""Content-family codecs for compressed tool results."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from tok.runtime.config import TOK_ENABLE_JSON_NONEXPANSION_GUARD, TOK_FORCE_FILE_CODEC
from tok.utils.token_utils import count_tokens

from ._tool_result_advisory import (
    _build_search_advisory,
    _estimate_tokens,
    _is_tok_cli_command,
)
from ._tool_result_file_read import _compress_file_read
from ._tool_result_truncation import truncate_large_result

__all__ = [
    "_compress_config_json",
    "_compress_env_ps",
    "_compress_file_read",
    "_compress_git_diff",
    "_compress_git_log",
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
    "_tighten_compressed_output",
    "truncate_large_result",
]

_CODE_PATTERNS = re.compile(r"\bdef \b|\bclass \b|\bimport \b|\basync def \b|\bfunction \b")


def _detect_tool_content_type(text: str, path: str = "") -> str:
    """Detect the content type of a tool result."""
    if path:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in {"py", "pyi"}:
            return "file"
    if "Traceback (most recent call last):" in text or "at new " in text:
        return "stack_trace"
    if re.search(r"\b(PASSED|FAILED)\b", text) and re.search(r"\d+ (passed|failed)( in | ,)", text):
        return "pytest"
    if re.search(r"^diff --git ", text, re.MULTILINE) or (
        re.search(r"^--- a/", text, re.MULTILINE) and re.search(r"^\+\+\+ b/", text, re.MULTILINE)
    ):
        return "git_diff"
    if re.match(r"^(USER\s+PID\s+%CPU|UID\s+PID\s+PPID)", text) or "COMMAND" in text[:200]:
        return "ps_output"
    if re.match(r"^(HOME|PATH|SHELL|USER|LANG)=", text, re.MULTILINE) and "=" in text:
        return "env_output"

    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]

    if len(non_empty) >= 4:
        grep_c_matches = sum(1 for line in non_empty if re.match(r"^[^\s-][^-]*-(\d+)-", line))
        if grep_c_matches / len(non_empty) > 0.6:
            return "grep_context"

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_skeleton"

    if len(non_empty) >= 2:
        if sum(1 for line in non_empty if _GIT_LOG_COMMIT_RE.match(line)) >= 2:
            return "git_log"
        oneline_matches = sum(1 for line in non_empty if _GIT_LOG_ONELINE_RE.match(line.strip()))
        if oneline_matches >= 4 and oneline_matches / len(non_empty) > 0.4:
            return "git_log"

    # Check grep path/line output before ls-like listings.
    if len(non_empty) >= 3:
        grep_matches = sum(
            1
            for line in non_empty
            if re.match(r"^\S+:\d+:", line)  # path:line:content
            or re.match(r"^\S+\.[A-Za-z0-9]{1,8}:[^\n]+$", line)  # file.ext:content
        )
        if grep_matches / len(non_empty) > 0.7:
            return "grep"
        bare_lnum_matches = sum(
            1
            for line in non_empty
            if re.match(r"^\d+:\s*\S", line)  # linenum:content (single-file grep -n)
        )
        if bare_lnum_matches / len(non_empty) > 0.7:
            return "grep"

    if len(non_empty) >= 8:
        la_lines = sum(1 for line in non_empty if re.match(r"^[dl-][rwx-]{9}", line))
        plain_file_lines = sum(
            1 for line in non_empty if re.match(r"^\S+\.\w{1,6}$", line.strip()) or re.match(r"^\S+/$", line.strip())
        )
        glob_lines = sum(1 for line in non_empty if re.match(r"^(/[^/ ]+)+$", line.strip()))
        if la_lines >= 6 or plain_file_lines / len(non_empty) > 0.7 or glob_lines / len(non_empty) > 0.7:
            return "ls"

    if len(non_empty) >= 6:
        install_lines = sum(1 for line in non_empty if _INSTALL_PROGRESS_RE.match(line))
        if install_lines >= 5:
            return "install"

    if len(text) > 1000 and _CODE_PATTERNS.search(text):
        _has_js_signals = bool(
            re.search(r"\b(const|let|var)\b", text)
            or re.search(r"=>\s*[{(]", text)
            or re.search(r"\bexport\s+(default\s+)?", text)
            or re.search(r"[{]\s*\.\.\.", text)
        )
        if not _has_js_signals:
            return "file"

    if TOK_FORCE_FILE_CODEC and len(text) > 200:
        if _CODE_PATTERNS.search(text) or text.count("\n") > 5:
            _has_js = bool(re.search(r"\b(const|let|var)\b", text) or re.search(r"=>\s*[{(]", text))
            if not _has_js:
                return "file"

    if len(lines) >= 5:
        for i in range(len(lines) - 4):
            prefix = re.split(r"[/: ]", lines[i].rstrip())[0]
            if prefix and all(lines[i + j].rstrip().startswith(prefix) for j in range(1, 5)):
                return "repetitive"

    return "raw"


def _compress_pytest(text: str, command: str = "") -> str:
    lines = text.splitlines()
    result: list[str] = []
    in_failure = False
    passed = 0
    failed = 0
    first_passed = ""
    first_failed = ""

    def _normalize_verification_command(command: str) -> str:
        cleaned = " ".join(command.split())
        if not cleaned:
            return ""
        return cleaned[:120]

    def _extract_failure_label(line: str) -> str:
        if not line:
            return ""
        label = line.strip()
        if label.endswith(" FAILED"):
            label = label[: -len(" FAILED")].strip()
        return label[:120]

    def _extract_pass_label(line: str) -> str:
        if not line:
            return ""
        label = line.strip()
        if label.endswith(" PASSED"):
            label = label[: -len(" PASSED")].strip()
        return label[:120]

    summary_passed = 0
    summary_failed = 0

    for line in lines:
        if re.match(r"=+\s+\d+.*\s+=+\s*$", line):
            result.append(line)
            in_failure = False
            if passed == 0 and failed == 0:
                m_passed = re.search(r"(\d+)\s+passed", line)
                if m_passed:
                    summary_passed = int(m_passed.group(1))
                m_failed = re.search(r"(\d+)\s+failed", line)
                if m_failed:
                    summary_failed = int(m_failed.group(1))
            continue
        if " PASSED" in line or line.endswith(" PASSED"):
            passed += 1
            if not first_passed:
                first_passed = _extract_pass_label(line)
            in_failure = False
            continue
        if " FAILED" in line or line.endswith(" FAILED"):
            failed += 1
            if not first_failed:
                first_failed = _extract_failure_label(line)
            in_failure = True
            result.append(line)
            continue
        if line.startswith(("=", "_")):
            in_failure = line.startswith("_ FAILURES") or "FAILED" in line or in_failure
            result.append(line)
            continue
        if in_failure:
            result.append(line)
            continue
        if line.startswith(("collected ", "platform ", "rootdir")):
            result.append(line)

    if passed == 0 and failed == 0:
        if summary_passed or summary_failed:
            passed = summary_passed
            failed = summary_failed

    header = f">>> tool:pytest|passed:{passed}|failed:{failed}"
    verification_line = ""
    command = _normalize_verification_command(command)
    if failed:
        failure_count = "1 failed" if failed == 1 else f"{failed} failed"
        if command and first_failed:
            verification_line = f"verification: {command} -> {first_failed} ({failure_count})"
        elif command:
            verification_line = f"verification: {command} ({failure_count})"
        elif first_failed:
            verification_line = f"verification: {first_failed} ({failure_count})"
        else:
            verification_line = f"verification: {failure_count}"
    elif passed:
        pass_count = "1 passed" if passed == 1 else f"{passed} passed"
        if command:
            verification_line = f"verification: {command} {pass_count}"
        elif first_passed:
            verification_line = f"verification: {first_passed} ({pass_count})"
        else:
            verification_line = f"verification: {pass_count}"

    parts = [header]
    if verification_line:
        parts.append(verification_line)
    if result:
        parts.append("\n".join(result))
    return "\n".join(parts)


def _compress_grep(text: str) -> str:
    lines = text.splitlines()
    by_file: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []

    for line in lines:
        m = re.match(r"^([^\s:][^:]*):(\d+):(.*)", line)
        if m:
            path, lnum, snippet = m.group(1), m.group(2), m.group(3)
        else:
            m_bare = re.match(r"^(\d+):(.*)", line)
            if m_bare:
                path, lnum, snippet = "", m_bare.group(1), m_bare.group(2)
            else:
                lnum = ""
                m2 = re.match(r"^([^\s:][^:]*):(.+)", line)
                if m2:
                    _candidate = m2.group(1)
                    if any(c in _candidate for c in ("/", ".", "\\")) or _candidate.startswith("~"):
                        path, snippet = m2.group(1), m2.group(2)
                    else:
                        path, snippet = "", line
                else:
                    path, snippet = "", line
        key = path or "__other__"
        if key not in by_file:
            by_file[key] = []
            order.append(key)
        by_file[key].append((lnum, snippet.strip()))

    total = sum(len(v) for v in by_file.values())
    if total <= 3:
        return text

    file_count = len([k for k in order if k != "__other__"])

    # Scale snippet limit based on total matches:
    # Small results (≤20) → show all; Medium (≤50) → 6/file; Large (>50) → 3/file
    if total <= 20:
        per_file_limit = 999  # effectively unlimited — show all
    elif total <= 50:
        per_file_limit = 6
    else:
        per_file_limit = 3

    result = [f">>> tool:grep|matches:{total}|files:{file_count}"]
    for key in order:
        snippets = by_file[key]
        limit = min(per_file_limit, len(snippets))
        shown = snippets[:limit]
        for lnum, s in shown:
            if lnum:
                result.append(f"{key}:{lnum}: {s[:80]}")
            else:
                result.append(f"{key}: {s[:80]}")
        remaining = len(snippets) - limit
        if remaining > 0:
            result.append(f"{key}: ... ({remaining} more matches)")

    compressed = "\n".join(result)
    # If compression doesn't save space, return original
    if len(compressed) >= len(text):
        return text
    return compressed


def _compress_repetitive(text: str, command: str = "") -> str:
    if _is_tok_cli_command(command):
        return text

    lines = text.splitlines()
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        parts = re.split(r"[/: ]", line.rstrip())
        prefix = next((p for p in parts if p), "")

        if prefix:
            j = i + 1
            while j < len(lines) and lines[j].rstrip().startswith(prefix):
                j += 1
            run_len = j - i
            if run_len >= 5:
                result.append(f"[{prefix}...]: {run_len} lines")
                i = j
                continue

        result.append(line)
        i += 1

    header = f">>> tool:bash|original_lines:{len(lines)}|compressed_lines:{len(result)}"
    compressed = header + "\n" + "\n".join(result)
    if len(compressed) >= len(text):
        return text
    return compressed


def _compress_git_diff(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    files = 0
    insertions = 0
    deletions = 0

    for line in lines:
        if line.startswith(("diff --git", "index ")):
            if line.startswith("diff --git"):
                files += 1
            result.append(line)
        elif line.startswith(("---", "+++", "@@")):
            result.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            insertions += 1
            result.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            result.append(line)
        elif not line.strip():
            result.append(line)

    if len(result) >= len(lines):
        return text

    header = f">>> tool:git_diff|files:{files}|insertions:{insertions}|deletions:{deletions}"
    return header + "\n" + "\n".join(result)


def _compress_ls(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    is_la = any(re.match(r"^(total\s+\d+|[dl-][rwx-]{9})", line) for line in lines)

    names: list[str] = []
    dirs: list[str] = []
    name_to_info: dict[str, str] = {}

    for line in lines:
        if is_la:
            parts = line.split()
            if not parts:
                continue
            if parts[0].startswith("total"):
                continue
            name = parts[-1]
            if line.startswith("d"):
                dirs.append(name)
            else:
                names.append(name)
                # If we have size info (typical ls -la), keep it
                if len(parts) >= 5:
                    size = parts[-5]
                    name_to_info[name] = size
        else:
            names.append(line.strip())

    ext_counts: dict[str, int] = {}
    unusual: list[str] = []
    for name in names:
        if "." in name and not name.startswith("."):
            ext = name.rsplit(".", 1)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        else:
            unusual.append(name)

    result_lines = [f">>> tool:ls|total:{len(names) + len(dirs)}|dirs:{len(dirs)}"]

    # If small directory, include actual filenames
    if len(names) > 0 and len(names) <= 20:
        file_list = []
        for n in names:
            info = name_to_info.get(n)
            file_list.append(f"{n} ({info})" if info else n)
        result_lines.append(f"  files: {', '.join(file_list)}")
    else:
        for ext, count in sorted(ext_counts.items(), key=lambda item: -item[1]):
            result_lines.append(f"  .{ext}: {count}")
        if unusual:
            result_lines.append(f"  other: {', '.join(unusual[:10])}" + (" ..." if len(unusual) > 10 else ""))

    if dirs:
        result_lines.append(f"  dirs: {', '.join(dirs[:10])}" + (" ..." if len(dirs) > 10 else ""))

    result = "\n".join(result_lines)
    if len(result) >= len(text):
        return text
    return result


_INSTALL_PROGRESS_RE = re.compile(
    r"^\s*(Downloading|Installing|Resolving|Fetching|Installed|Resolved|Locked"
    r"|Preparing|Collecting|Obtaining|Already satisfied|Using cached"
    r"|Requirement already|Building|Running|Successfully installed"
    r"|Prepared|Uninstalled|Built)",
    re.IGNORECASE,
)
_INSTALL_ERROR_RE = re.compile(r"\b(error|warning|failed|conflict)\b", re.IGNORECASE)
_INSTALL_SUMMARY_RE = re.compile(
    r"(Successfully installed|installed \d+|added \d+|in \d+\.\d+s|\d+ packages?)",
    re.IGNORECASE,
)
_INSTALL_FAILURE_SIGNAL_RE = re.compile(
    r"(\berror\b|\bfailed\b|traceback|exception|npm err!|pip subprocess|could not build wheels)",
    re.IGNORECASE,
)


def _compress_install(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    summary_line = ""
    packages = 0
    duration = ""
    failure_index: int | None = None

    for idx, line in enumerate(lines):
        if failure_index is None and _INSTALL_FAILURE_SIGNAL_RE.search(line):
            failure_index = idx
        if _INSTALL_SUMMARY_RE.search(line):
            summary_line = line
            match = re.search(r"in\s+(\d+\.\d+s)", line)
            if match:
                duration = match.group(1)
            continue
        if _INSTALL_ERROR_RE.search(line):
            kept.append(line)
            continue
        if _INSTALL_PROGRESS_RE.match(line):
            packages += 1
            continue
        kept.append(line)

    if failure_index is not None:
        start = max(0, failure_index - 2)
        failure_tail = lines[start:]
        header = f">>> tool:install|packages:{packages}|duration:{duration or 'unknown'}|status:failed"
        result = header + "\n" + "\n".join(failure_tail)
        return result if len(result) < len(text) else text

    if summary_line:
        kept.append(summary_line)

    header = f">>> tool:install|packages:{packages}|duration:{duration or 'unknown'}"
    result = header + "\n" + "\n".join(kept)
    if len(result) >= len(text):
        return text
    return result


_GIT_LOG_COMMIT_RE = re.compile(r"^commit ([0-9a-f]{40})$")
_GIT_LOG_ONELINE_RE = re.compile(r"^([0-9a-f]{7,40})\s+(.+)")


def _compress_git_log(text: str) -> str:
    lines = text.splitlines()

    oneline = all(not line.strip() or _GIT_LOG_ONELINE_RE.match(line) for line in lines if line.strip())
    if oneline:
        entries: list[str] = []
        for line in lines:
            match = _GIT_LOG_ONELINE_RE.match(line.strip())
            if match:
                entries.append(f"{match.group(1)[:8]} {match.group(2)[:80]}")
        if not entries:
            return text
        header = f">>> tool:git_log|commits:{len(entries)}"
        result = header + "\n" + "\n".join(entries)
        if len(result) >= len(text):
            return text
        return result

    entries = []
    current: dict[str, str] = {}
    in_body = False

    for line in lines:
        match = _GIT_LOG_COMMIT_RE.match(line)
        if match:
            if current.get("hash"):
                entries.append(f"{current.get('hash', '')} {current.get('subject', '')[:40]}")
            current = {
                "hash": match.group(1)[:8],
                "author": "",
                "date": "",
                "subject": "",
            }
            in_body = False
            continue
        if line.startswith("Author:"):
            parts = line[7:].strip().split("<")[0].strip().split()
            current["author"] = parts[0] if parts else ""
            in_body = False
            continue
        if line.startswith("Date:"):
            current["date"] = line[5:].strip()[:20]
            in_body = False
            continue
        stripped = line.strip()
        if stripped and not in_body and current.get("hash") and not current["subject"]:
            current["subject"] = stripped[:72]
            in_body = True

    if current.get("hash"):
        entries.append(f"{current.get('hash', '')} {current.get('subject', '')[:40]}")

    if not entries:
        return text

    result_lines = [f">>> tool:git_log|commits:{len(entries)}"]
    result_lines.extend(entries)
    result = "\n".join(result_lines)
    if len(result) >= len(text):
        return text
    return result


def _compress_search_results(text: str) -> str:
    try:
        data = json.loads(text)
        if not isinstance(data, list) or not data:
            return text

        sample = data[0]
        if not isinstance(sample, dict):
            return text

        common_keys = [key for key in sample if all(key in item for item in data[:5])]
        evidence_keys = {
            key for key in common_keys if key in {"line", "snippet", "content", "text", "match", "context"}
        }
        if not evidence_keys:
            return text

        header_keys = [key for key in ("path", "file", "name", "title", "line", "id") if key in common_keys]
        if not header_keys:
            header_keys = common_keys[:3]

        value_keys = [
            key
            for key in (
                "path",
                "file",
                "name",
                "line",
                "snippet",
                "text",
                "match",
                "context",
                "id",
            )
            if key in common_keys
        ]
        if not value_keys:
            return text

        result_count = len(data)
        result = [f">>> tool:search_results|count:{result_count}|keys:{','.join(header_keys)}"]
        for item in data:
            vals = [str(item.get(key, ""))[:80].replace("\n", " ") for key in value_keys]
            if not any(val.strip() for val in vals):
                continue
            result.append(":".join(vals))

        # Add advisory footer for large result sets
        # Use file_count=0 since we don't have per-file breakdown in JSON results
        estimated_tokens = _estimate_tokens(text)
        advisory = _build_search_advisory(
            match_count=result_count,
            file_count=0,
            estimated_tokens=estimated_tokens,
            has_scope=True,  # JSON results typically come from scoped searches
        )
        if advisory:
            result.append(advisory)

        return "\n".join(result)
    except Exception:
        return text


def _compress_stack_traces(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    lib_patterns = re.compile(
        r"(node_modules|site-packages|dist-packages|/lib/python|/usr/lib|/usr/include|/Library/Frameworks|/usr/local/Cellar)"
    )

    paths = re.findall(r'File "([^"]+)"', text)
    common_prefix = ""
    if len(paths) >= 2:
        try:
            common_prefix = os.path.commonpath(paths) if hasattr(os, "commonpath") else ""
            if common_prefix and len(common_prefix) < 10:
                common_prefix = ""
        except ValueError:
            # Paths have no common prefix or are on different drives
            common_prefix = ""

    hidden_count = 0
    for line in lines:
        match = re.search(r'File "(.+)", line (\d+), in (\w+)', line)
        if match:
            path, line_num, func = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if lib_patterns.search(path):
                hidden_count += 1
                continue
            if common_prefix and path.startswith(common_prefix):
                path = "..." + path[len(common_prefix) :]
            result.append(f"  at {func} ({path}:{line_num})")
            continue

        match = re.search(r"at (\w+) \((.+):(\d+):(\d+)\)", line)
        if match:
            func, path, lnum, _col = (
                match.group(1),
                match.group(2),
                match.group(3),
                match.group(4),
            )
            if lib_patterns.search(path):
                hidden_count += 1
                continue
            if common_prefix and path.startswith(common_prefix):
                path = "..." + path[len(common_prefix) :]
            result.append(f"  at {func} ({path}:{lnum})")
            continue

        result.append(line)

    if hidden_count > 0:
        result.insert(0, f"  [... filtered {hidden_count} library frames]")

    header = f">>> tool:stack_trace|lines:{len(lines)}|hidden_frames:{hidden_count}"
    compressed = header + "\n" + "\n".join(result)
    if len(compressed) >= len(text):
        return text
    return compressed


def _compress_json_response(data: str | dict[str, Any] | list[Any], depth: int = 0) -> str | dict[str, Any] | list[Any]:
    if isinstance(data, dict):
        if len(data) > 20 and depth > 1:
            return f"{{... {len(data)} keys}}"
        return {key: _compress_json_response(value, depth + 1) for key, value in data.items()}
    if isinstance(data, list):
        if len(data) > 10:
            return [
                _compress_json_response(data[0], depth + 1),
                f"... {len(data) - 1} more items",
            ]
        return [_compress_json_response(item, depth + 1) for item in data]
    if isinstance(data, str) and len(data) > 200:
        return data[:197] + "..."
    return data


def _compress_grep_context(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result = []
    current_file = None
    current_block: list[str] = []
    last_line_num = -1

    for line in lines:
        match = re.match(r"^([^\s-][^-]*)-(\d+)-(.*)", line)
        if match:
            path, lnum, content = (
                match.group(1),
                int(match.group(2)),
                match.group(3),
            )
            if path != current_file:
                if current_block:
                    result.append(f"  [{last_line_num}]")
                current_file = path
                result.append(f"file://{path}:")
                result.append(f"  [{lnum}] {content}")
                last_line_num = lnum
            else:
                if lnum > last_line_num + 1:
                    result.append("  ...")
                result.append(f"  [{lnum}] {content}")
                last_line_num = lnum
            continue

        match = re.match(r"^([^\s:][^:]*):(\d+):(.*)", line)
        if match:
            path, lnum, content = (
                match.group(1),
                int(match.group(2)),
                match.group(3),
            )
            if path != current_file:
                current_file = path
                result.append(f"file://{path}:")
            result.append(f"  [{lnum}]* {content}")
            last_line_num = lnum
            continue

        result.append(line)

    line_count = len(lines)
    header = f">>> tool:grep_context|lines:{line_count}"
    output = header + "\n" + "\n".join(result)

    # Add advisory footer for large context results
    # Use match_count=line_count since we don't have separate match/file counts
    estimated_tokens = _estimate_tokens(text)
    advisory = _build_search_advisory(
        match_count=line_count,
        file_count=0,
        estimated_tokens=estimated_tokens,
        has_scope=True,  # Context results are typically scoped
    )
    if advisory:
        output = output + "\n" + advisory

    return output


def _compress_env_ps(text: str, kind: str) -> str:
    lines = text.splitlines()

    if kind == "ps_output":
        kept = [lines[0]] if lines else []
        for line in lines[1:]:
            if "/System/" in line or "/usr/libexec/" in line or "kernel_task" in line:
                continue
            kept.append(line)

        if len(kept) > 20:
            kept = [*kept[:20], f"... {len(kept) - 20} more active processes"]

        header = f">>> tool:ps|total_lines:{len(lines)}|interesting:{len(kept) - 1}"
        return header + "\n" + "\n".join(kept)

    if kind == "env_output":
        interesting = {
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "EDITOR",
            "LANG",
            "PWD",
            "VIRTUAL_ENV",
        }
        kept = []
        for line in lines:
            if "=" in line:
                key = line.split("=", 1)[0]
                if key in interesting or "API" in key or "TOKEN" in key or "URL" in key or "PORT" in key:
                    kept.append(line)

        header = f">>> tool:env|total_vars:{len(lines)}|displayed:{len(kept)}"
        return header + "\n" + "\n".join(kept)

    return text


def _compress_config_json(text: str) -> str:
    try:
        data = json.loads(text)
        skeleton = _compress_json_response(data)
        compressed = json.dumps(skeleton, indent=2)

        header = f">>> tool:json_skeleton|original_chars:{len(text)}|saved_chars:{len(text) - len(compressed)}"
        candidate = header + "\n" + compressed
        if TOK_ENABLE_JSON_NONEXPANSION_GUARD and count_tokens(candidate) >= count_tokens(text):
            return text
        return candidate if len(candidate) < len(text) else text
    except Exception:
        return text


def _tighten_compressed_output(kind: str, compressed: str, compression_level: str) -> str:
    if compression_level != "aggressive":
        return compressed
    if kind not in {
        "grep",
        "grep_context",
        "ls",
        "install",
        "repetitive",
        "search_results",
    }:
        return compressed
    lines = compressed.splitlines()
    if len(lines) <= 4:
        return compressed
    header = lines[0]
    body = lines[1:]
    limit = 4
    if len(body) <= limit:
        return compressed
    trimmed = [
        header,
        *body[:limit],
        f"... {len(body) - limit} more lines omitted",
    ]
    candidate = "\n".join(trimmed)
    return candidate if len(candidate) < len(compressed) else compressed
