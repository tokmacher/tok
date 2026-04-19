"""Truncation helpers for extremely large tool results."""

from __future__ import annotations

import re


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

    return min(range(1, len(lines)), key=lambda idx: abs(offsets[idx] - target_chars))


_PYTEST_SECTION_RE = re.compile(r"^[=_\-]{3,}\s*([A-Z]+[ _][A-Z]+|[A-Z]+)\s*[=_\-]{3,}\s*$")
_PYTEST_SUMMARY_RE = re.compile(
    r"^[=_\-]{3,}\s*\d+.*(?:passed|failed|error|warning|skipped).*\s*[=_\-]{3,}\s*$", re.IGNORECASE | re.MULTILINE
)


def _pytest_aware_truncation(text: str, lines: list[str], limit: int) -> str | None:
    if not _PYTEST_SUMMARY_RE.search(text):
        return None

    section_starts: list[int] = []
    summary_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _PYTEST_SUMMARY_RE.match(stripped):
            summary_idx = i
        elif _PYTEST_SECTION_RE.match(stripped):
            section_starts.append(i)

    failures_start: int | None = None
    for idx in section_starts:
        if "FAIL" in lines[idx].upper():
            failures_start = idx
            break

    if failures_start is None and summary_idx is None:
        return None

    head_end = min(3, len(lines))
    if failures_start is not None:
        head_end = max(head_end, failures_start)

    kept_indices: set[int] = set(range(head_end))
    if summary_idx is not None:
        for j in range(summary_idx, len(lines)):
            kept_indices.add(j)
    if failures_start is not None:
        failures_end = min(failures_start + 80, len(lines))
        for j in range(failures_start, failures_end):
            kept_indices.add(j)

    ordered = sorted(kept_indices)
    out_parts: list[str] = []
    prev = -1
    omitted_lines = 0
    for idx in ordered:
        if idx > prev + 1:
            omitted_lines += idx - prev - 1
            out_parts.append(f"[{idx - prev - 1} lines omitted]\n")
        out_parts.append(lines[idx])
        prev = idx

    original_chars = len(text)
    result = "".join(out_parts)
    omitted_chars = original_chars - len(result)
    if len(result) >= original_chars:
        return None

    marker = (
        f"\n... [TRUNCATED {omitted_chars} CHARS; {omitted_lines} lines omitted; FAILURES + summary preserved] ...\n"
    )
    insert_pos = min(head_end, len(out_parts))
    parts = ["".join(out_parts[:insert_pos]), "".join(out_parts[insert_pos:])]
    return parts[0] + marker + parts[1]


def _extract_symbol_table(lines: list[str], start: int, end: int) -> str:
    """
    Scan lines[start:end] for top-level Python symbols and return a compact
    @symbols block with line numbers.

    Only emits imports at the top of the omitted section, plus every class/def
    whose indentation level is 0 or 4 (i.e. top-level and first-level methods).
    Returns an empty string when no symbols are found (e.g. data/log output).
    """
    _import_re = re.compile(r"^(import |from \S+ import )")
    _symbol_re = re.compile(r"^(\s*)(class |def |async def )(\w+)")
    _MAX_SYMBOLS = 40

    entries: list[str] = []

    for rel_idx, raw in enumerate(lines[start:end]):
        abs_line = start + rel_idx + 1  # 1-based line number for the reader
        stripped = raw.rstrip()

        # Gather leading imports compactly (first block only, max 5)
        if _import_re.match(stripped):
            if len([e for e in entries if "import" in e]) < 5:
                entries.append(f"  |> L{abs_line:<5} {stripped[:80]}")
            continue

        m = _symbol_re.match(stripped)
        if not m:
            continue
        indent = len(m.group(1))
        if indent > 8:  # skip deeply nested helpers
            continue
        sig = stripped.strip()
        if len(sig) > 72:
            sig = sig[:69] + "..."
        entries.append(f"  |> L{abs_line:<5} {sig}")
        if len(entries) >= _MAX_SYMBOLS:
            entries.append(f"  |> ... ({end - start - rel_idx} more lines not shown)")
            break

    if not entries:
        return ""
    return "@symbols\n" + "\n".join(entries) + "\n"


def truncate_large_result(text: str, limit: int = 1200, *, already_compressed: bool = False) -> str:
    # If already compressed (e.g., skeletonized by file_read), don't truncate further.
    if already_compressed:
        return text

    if len(text) <= int(limit * 1.5):
        return text

    # Don't truncate small multi-line files at default limits - they're high-value discovery targets.
    # But still truncate if caller explicitly requests a smaller limit.
    line_count = text.count("\n") + 1
    avg_line_len = len(text) / max(1, line_count)
    if limit >= 1000 and line_count >= 2 and line_count < 100 and avg_line_len < 100:
        return text

    lines = text.splitlines(keepends=True)

    pytest_result = _pytest_aware_truncation(text, lines, limit)
    if pytest_result is not None:
        return pytest_result

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
                important_line = f"\n... [SIGNAL FOUND] {line.strip()[:100]} ..."
                break

        omitted = len(text) - (limit // 2 * 2)
        return f"{head}\n... [TRUNCATED {omitted} CHARS] ...{important_line}\n{tail}"

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    search_window_chars = max(80, limit // 4)
    head_target = max(1, limit // 2)
    tail_target = max(head_target + 1, len(text) - (limit // 2))

    head_idx = _choose_line_boundary(lines, offsets, head_target, search_window_chars)
    tail_idx = _choose_line_boundary(lines, offsets, tail_target, search_window_chars)
    if head_idx >= tail_idx:
        tail_idx = min(len(lines), max(head_idx + 1, tail_idx))
    if head_idx >= tail_idx:
        head_idx = max(1, len(lines) // 2)
        tail_idx = min(len(lines), head_idx + 1)

    head_text = "".join(lines[:head_idx])
    tail_text = "".join(lines[tail_idx:])
    omitted_chars = max(0, len(text) - len(head_text) - len(tail_text))
    continuation_line = tail_idx + 1 if tail_idx < len(lines) else len(lines)
    symbol_table = _extract_symbol_table(lines, head_idx, tail_idx)
    marker = (
        f"... [TRUNCATED {omitted_chars} CHARS; omitted lines "
        f"{head_idx + 1}-{tail_idx}; continue at line {continuation_line}]\n"
        f"{symbol_table}"
        f"... [use offset={head_idx + 1} to read omitted section] ..."
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

    # Fallback: if we failed to shorten, use a signal-preserving head/tail truncation.
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
    return f"{head}\n... [TRUNCATED {omitted} CHARS] ...{important_line}\n{tail}"
