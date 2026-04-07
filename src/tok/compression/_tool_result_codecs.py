from __future__ import annotations

"""Content-family codecs for compressed tool results."""

import json
import os
import re
from typing import Any

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

_CODE_PATTERNS = re.compile(
    r"\bdef \b|\bclass \b|\bimport \b|\basync def \b|\bfunction \b"
)


def _detect_tool_content_type(text: str) -> str:
    """Detect the content type of a tool result."""
    if "Traceback (most recent call last):" in text or "at new " in text:
        return "stack_trace"
    if re.search(r"\b(PASSED|FAILED)\b", text) and re.search(
        r"\d+ (passed|failed)( in | ,)", text
    ):
        return "pytest"
    if re.search(r"^diff --git ", text, re.MULTILINE) or (
        re.search(r"^--- a/", text, re.MULTILINE)
        and re.search(r"^\+\+\+ b/", text, re.MULTILINE)
    ):
        return "git_diff"
    if (
        re.match(r"^(USER\s+PID\s+%CPU|UID\s+PID\s+PPID)", text)
        or "COMMAND" in text[:200]
    ):
        return "ps_output"
    if (
        re.match(r"^(HOME|PATH|SHELL|USER|LANG)=", text, re.MULTILINE)
        and "=" in text
    ):
        return "env_output"

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]

    if len(non_empty) >= 4:
        grep_c_matches = sum(
            1 for l in non_empty if re.match(r"^[^\s-][^-]*-(\d+)-", l)
        )
        if grep_c_matches / len(non_empty) > 0.6:
            return "grep_context"

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_skeleton"

    if len(non_empty) >= 2:
        if sum(1 for l in non_empty if _GIT_LOG_COMMIT_RE.match(l)) >= 2:
            return "git_log"
        oneline_matches = sum(
            1 for l in non_empty if _GIT_LOG_ONELINE_RE.match(l.strip())
        )
        if oneline_matches >= 4 and oneline_matches / len(non_empty) > 0.4:
            return "git_log"

    if len(non_empty) >= 8:
        la_lines = sum(1 for l in non_empty if re.match(r"^[dl-][rwx-]{9}", l))
        plain_file_lines = sum(
            1
            for l in non_empty
            if re.match(r"^\S+\.\w{1,6}$", l.strip())
            or re.match(r"^\S+/$", l.strip())
        )
        glob_lines = sum(
            1 for l in non_empty if re.match(r"^(/[^/ ]+)+$", l.strip())
        )
        if (
            la_lines >= 6
            or plain_file_lines / len(non_empty) > 0.7
            or glob_lines / len(non_empty) > 0.7
        ):
            return "ls"

    if len(non_empty) >= 6:
        install_lines = sum(
            1 for l in non_empty if _INSTALL_PROGRESS_RE.match(l)
        )
        if install_lines >= 5:
            return "install"

    if len(non_empty) >= 3:
        grep_matches = sum(
            1
            for l in non_empty
            if re.match(r"^[^\s:][^:]*:\d+:", l)
            or re.match(r"^[^\s:][^:]*:[^\n]+$", l)
        )
        if grep_matches / len(non_empty) > 0.7:
            return "grep"

    if len(text) > 1000 and _CODE_PATTERNS.search(text):
        return "file"

    if len(lines) >= 5:
        for i in range(len(lines) - 4):
            prefix = re.split(r"[/: ]", lines[i].rstrip())[0]
            if prefix and all(
                lines[i + j].rstrip().startswith(prefix) for j in range(1, 5)
            ):
                return "repetitive"

    if text.strip().startswith("[") and text.strip().endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) >= 3:
                if all(isinstance(x, dict) for x in data[:3]):
                    return "search_results"
        except Exception:
            pass

    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and len(data) >= 5:
                return "config_json"
        except Exception:
            pass

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

    for line in lines:
        if re.match(r"=+\s+\d+.*\s+=+\s*$", line):
            result.append(line)
            in_failure = False
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
        if line.startswith("=") or line.startswith("_"):
            in_failure = (
                line.startswith("_ FAILURES") or "FAILED" in line or in_failure
            )
            result.append(line)
            continue
        if in_failure:
            result.append(line)
        elif (
            line.startswith("collected ")
            or line.startswith("platform ")
            or line.startswith("rootdir")
        ):
            result.append(line)

    header = f">>> tool:pytest|passed:{passed}|failed:{failed}"
    verification_line = ""
    command = _normalize_verification_command(command)
    if failed:
        failure_count = "1 failed" if failed == 1 else f"{failed} failed"
        if command and first_failed:
            verification_line = (
                f"verification: {command} -> {first_failed} ({failure_count})"
            )
        elif command:
            verification_line = f"verification: {command} ({failure_count})"
        elif first_failed:
            verification_line = (
                f"verification: {first_failed} ({failure_count})"
            )
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
    by_file: dict[str, list[str]] = {}
    order: list[str] = []

    for line in lines:
        m = re.match(r"^([^\s:][^:]*):(\d+):(.*)", line)
        if m:
            path, _lnum, snippet = m.group(1), m.group(2), m.group(3)
        else:
            m2 = re.match(r"^([^\s:][^:]*):(.+)", line)
            if m2:
                path, snippet = m2.group(1), m2.group(2)
            else:
                path, snippet = "", line
        key = path or "__other__"
        if key not in by_file:
            by_file[key] = []
            order.append(key)
        by_file[key].append(snippet.strip())

    total = sum(len(v) for v in by_file.values())
    if total <= 3:
        return text

    result = [
        f">>> tool:grep|matches:{total}|files:{len([k for k in order if k != '__other__'])}"
    ]
    for key in order:
        snippets = by_file[key]
        first = snippets[0][:80]
        suffix = f" ({len(snippets)} matches)" if len(snippets) > 1 else ""
        result.append(
            f"{key}: {len(snippets)} match{'es' if len(snippets) > 1 else ''} — {first}{suffix}"
        )

    return "\n".join(result)


def _compress_repetitive(text: str) -> str:
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

    if len(result) >= len(lines):
        return text

    header = f">>> tool:bash|original_lines:{len(lines)}|compressed_lines:{len(result)}"
    return header + "\n" + "\n".join(result)


def _compress_file_read(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    i = 0
    in_body = False
    body_line_count = 0

    signature_re = re.compile(
        r"^(import |from |class |def |async def |[A-Z_][A-Z0-9_]+ =|\s*def |\s*async def |\s*class )"
    )

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            result.append("")
            i += 1
            continue

        if signature_re.match(line):
            if in_body and body_line_count > 0:
                result.append(f"  |> [{body_line_count} lines]")
            in_body = False
            body_line_count = 0
            result.append(line)
            i += 1
            continue

        if re.match(r"^\s+(def |async def |class )", line):
            if in_body and body_line_count > 0:
                result.append(f"  |> [{body_line_count} lines]")
            in_body = False
            body_line_count = 0
            result.append(line)
            i += 1
            continue

        if in_body:
            body_line_count += 1
        else:
            in_body = True
            body_line_count = 1
        i += 1

    if in_body and body_line_count > 0:
        result.append(f"  |> [{body_line_count} lines]")

    if len(result) >= len(lines):
        return text

    trimmed_result = result
    if len(result) > 32:
        head_count = 18
        tail_count = 8
        omitted = max(0, len(result) - head_count - tail_count)
        trimmed_result = list(result[:head_count])
        if omitted:
            trimmed_result.append(f"  |> [{omitted} skeleton lines omitted]")
        trimmed_result.extend(result[-tail_count:])

    original_chars = len(text)
    compressed = "\n".join(trimmed_result)
    header = (
        f">>> tool:file_read|original_chars:{original_chars}|"
        f"skeleton_lines:{len(result)}|retained_skeleton_lines:{len(trimmed_result)}"
    )
    return header + "\n" + compressed


def _compress_git_diff(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    files = 0
    insertions = 0
    deletions = 0

    for line in lines:
        if line.startswith("diff --git") or line.startswith("index "):
            if line.startswith("diff --git"):
                files += 1
            result.append(line)
        elif line.startswith("---") or line.startswith("+++"):
            result.append(line)
        elif line.startswith("@@"):
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


def _leading_whitespace_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _line_boundary_priority(prev_line: str, next_line: str) -> int:
    prev_stripped = prev_line.strip()
    next_stripped = next_line.strip()
    if not prev_stripped or not next_stripped:
        return 0

    prev_indent = _leading_whitespace_width(prev_line)
    next_indent = _leading_whitespace_width(next_line)
    if next_indent < prev_indent:
        return 0

    if re.match(
        r"^\s*(?:@|def |class |async def |if |for |while |with |try\b|except\b|elif\b|else\b|match\b|case\b)",
        next_line,
    ):
        return 1

    return 2


def _choose_line_boundary(
    lines: list[str],
    offsets: list[int],
    target_chars: int,
    search_window_chars: int,
) -> int:
    if len(lines) <= 1:
        return len(lines)

    candidates: list[tuple[int, int, int]] = []
    for idx in range(1, len(lines)):
        distance = abs(offsets[idx] - target_chars)
        if distance > search_window_chars:
            continue
        candidates.append(
            (
                _line_boundary_priority(lines[idx - 1], lines[idx]),
                distance,
                idx,
            )
        )

    if candidates:
        return min(candidates)[2]

    return min(
        range(1, len(lines)), key=lambda idx: abs(offsets[idx] - target_chars)
    )


def _compress_ls(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    is_la = any(
        re.match(r"^(total\s+\d+|[dl-][rwx-]{9})", line) for line in lines
    )

    names: list[str] = []
    dirs: list[str] = []

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

    result_lines = [
        f">>> tool:ls|total:{len(names) + len(dirs)}|dirs:{len(dirs)}"
    ]
    for ext, count in sorted(ext_counts.items(), key=lambda item: -item[1]):
        result_lines.append(f"  .{ext}: {count}")
    if dirs:
        result_lines.append(
            f"  dirs: {', '.join(dirs[:10])}"
            + (" ..." if len(dirs) > 10 else "")
        )
    if unusual:
        result_lines.append(
            f"  other: {', '.join(unusual[:10])}"
            + (" ..." if len(unusual) > 10 else "")
        )

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
_INSTALL_ERROR_RE = re.compile(
    r"\b(error|warning|failed|conflict)\b", re.IGNORECASE
)
_INSTALL_SUMMARY_RE = re.compile(
    r"(Successfully installed|installed \d+|added \d+|in \d+\.\d+s|\d+ packages?)",
    re.IGNORECASE,
)


def _compress_install(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    summary_line = ""
    packages = 0
    duration = ""

    for line in lines:
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

    oneline = all(
        not line.strip() or _GIT_LOG_ONELINE_RE.match(line)
        for line in lines
        if line.strip()
    )
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
                entries.append(
                    f"{current.get('hash', '')} {current.get('subject', '')[:40]}"
                )
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
        if (
            stripped
            and not in_body
            and current.get("hash")
            and not current["subject"]
        ):
            current["subject"] = stripped[:72]
            in_body = True

    if current.get("hash"):
        entries.append(
            f"{current.get('hash', '')} {current.get('subject', '')[:40]}"
        )

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

        common_keys = [
            key
            for key in sample.keys()
            if all(key in item for item in data[:5])
        ]
        header_keys = [
            key
            for key in ("path", "file", "name", "title", "line", "id")
            if key in common_keys
        ]
        if not header_keys:
            header_keys = common_keys[:3]

        result = [
            f">>> tool:search_results|count:{len(data)}|keys:{','.join(header_keys)}"
        ]
        for item in data:
            vals = [
                str(item.get(key, ""))[:50].replace("\n", " ")
                for key in header_keys
            ]
            result.append(" | ".join(vals))

        return "\n".join(result)
    except Exception:
        return text


def _compress_stack_traces(text: str) -> str:
    lines = text.splitlines()
    result = []
    lib_patterns = re.compile(
        r"(node_modules|site-packages|dist-packages|/lib/python|/usr/lib|/usr/include|/Library/Frameworks|/usr/local/Cellar)"
    )

    paths = re.findall(r'File "([^"]+)"', text)
    common_prefix = ""
    if len(paths) >= 2:
        try:
            common_prefix = (
                os.path.commonpath(paths) if hasattr(os, "commonpath") else ""
            )
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

    header = (
        f">>> tool:stack_trace|lines:{len(lines)}|hidden_frames:{hidden_count}"
    )
    return header + "\n" + "\n".join(result)


def _compress_json_response(data: Any, depth: int = 0) -> Any:
    if isinstance(data, dict):
        if len(data) > 20 and depth > 1:
            return f"{{... {len(data)} keys}}"
        return {
            key: _compress_json_response(value, depth + 1)
            for key, value in data.items()
        }
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

    header = f">>> tool:grep_context|lines:{len(lines)}"
    return header + "\n" + "\n".join(result)


def _compress_env_ps(text: str, kind: str) -> str:
    lines = text.splitlines()

    if kind == "ps_output":
        kept = [lines[0]] if lines else []
        for line in lines[1:]:
            if (
                "/System/" in line
                or "/usr/libexec/" in line
                or "kernel_task" in line
            ):
                continue
            kept.append(line)

        if len(kept) > 20:
            kept = kept[:20] + [f"... {len(kept) - 20} more active processes"]

        header = (
            f">>> tool:ps|total_lines:{len(lines)}|interesting:{len(kept) - 1}"
        )
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
                if (
                    key in interesting
                    or "API" in key
                    or "TOKEN" in key
                    or "URL" in key
                    or "PORT" in key
                ):
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
        return header + "\n" + compressed
    except Exception:
        return text


def _tighten_compressed_output(
    kind: str, compressed: str, compression_level: str
) -> str:
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
    trimmed = (
        [header]
        + body[:limit]
        + [f"... {len(body) - limit} more lines omitted"]
    )
    candidate = "\n".join(trimmed)
    return candidate if len(candidate) < len(compressed) else compressed


def truncate_large_result(text: str, limit: int = 1200) -> str:
    if len(text) <= int(limit * 1.5):
        return text

    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        signals = re.compile(
            r"\b(error|fail|exception|traceback|parse_error|collision|conflict|issue|bug|diff|warning)\b",
            re.IGNORECASE,
        )

        head = text[: limit // 2]
        tail = text[-limit // 2 :]
        middle = text[limit // 2 : -limit // 2]

        important_line = ""
        for line in middle.splitlines():
            if signals.search(line):
                important_line = (
                    f"\n... [SIGNAL FOUND] {line.strip()[:100]} ..."
                )
                break

        omitted = len(text) - (limit // 2 * 2)
        return f"{head}\n... [TRUNCATED {omitted} CHARS] ...{important_line}\n{tail}"

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    search_window_chars = max(80, limit // 4)
    head_target = max(1, limit // 2)
    tail_target = max(head_target + 1, len(text) - (limit // 2))

    head_idx = _choose_line_boundary(
        lines, offsets, head_target, search_window_chars
    )
    tail_idx = _choose_line_boundary(
        lines, offsets, tail_target, search_window_chars
    )
    if head_idx >= tail_idx:
        tail_idx = min(len(lines), max(head_idx + 1, tail_idx))
    if head_idx >= tail_idx:
        head_idx = max(1, len(lines) // 2)
        tail_idx = min(len(lines), head_idx + 1)

    head_text = "".join(lines[:head_idx])
    tail_text = "".join(lines[tail_idx:])
    omitted_chars = max(0, len(text) - len(head_text) - len(tail_text))
    continuation_line = tail_idx + 1 if tail_idx < len(lines) else len(lines)
    marker = (
        f"... [TRUNCATED {omitted_chars} CHARS; omitted lines "
        f"{head_idx + 1}-{tail_idx}; continue at line {continuation_line}] ..."
    )

    compressed = head_text
    if compressed and not compressed.endswith("\n"):
        compressed += "\n"
    compressed += marker
    if tail_text:
        if not tail_text.startswith("\n"):
            compressed += "\n"
        compressed += tail_text

    if len(compressed) < len(text):
        return compressed

    signals = re.compile(
        r"\b(error|fail|exception|traceback|parse_error|collision|conflict|issue|bug|diff|warning)\b",
        re.IGNORECASE,
    )

    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    middle = text[limit // 2 : -limit // 2]

    important_line = ""
    for line in middle.splitlines():
        if signals.search(line):
            important_line = f"\n... [SIGNAL FOUND] {line.strip()[:100]} ..."
            break

    omitted = len(text) - (limit // 2 * 2)
    return (
        f"{head}\n... [TRUNCATED {omitted} CHARS] ...{important_line}\n{tail}"
    )
