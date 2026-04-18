"""Content-family codecs for compressed tool results."""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

from tok.runtime.config import TOK_ENABLE_JSON_NONEXPANSION_GUARD, TOK_FORCE_FILE_CODEC
from tok.utils.token_utils import count_tokens

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

# Thresholds for search-cost advisory
_GREP_ADVISORY_MATCH_THRESHOLD = 50
_GREP_ADVISORY_FILE_THRESHOLD = 10
_GREP_ADVISORY_TOKEN_THRESHOLD = 2000  # Estimated tokens

# Advisory cooldown state (per-query identity -> last advisory turn)
# This is a module-level cache that gets cleared between sessions
_advisory_cooldown: dict[str, int] = {}
_ADVISORY_COOLDOWN_TURNS = 3
_TOK_CLI_TOKEN_RE = re.compile(r"(?<![\w./-])tok(?![\w-])")


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

    The advisory is purely informational - it does not affect evidence policy
    or compression behavior. It simply alerts the model to consider narrowing
    scope on large result sets.

    Args:
        match_count: Number of matches found
        file_count: Number of files with matches
        estimated_tokens: Estimated token count of the result
        has_scope: Whether the search had path/glob/type restrictions
        query_identity: Canonical query identity for cooldown
        current_turn: Current turn number for cooldown tracking

    Returns:
        Advisory footer string, or empty string if no advisory warranted.

    """
    # Check cooldown first
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

    # Build specialized advisory based on conditions
    hints = []

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
        # Fallback for unscoped without other triggers
        hints.append("broad search")
        hints.append("consider narrowing scope")

    advisory = f"[tok advisory: {' - '.join(hints)}]"

    # Record in cooldown
    if query_identity:
        _advisory_cooldown[query_identity] = current_turn

    return advisory


def clear_advisory_cooldown() -> None:
    """Clear the advisory cooldown cache. Call between sessions."""
    global _advisory_cooldown
    _advisory_cooldown = {}


def _detect_tool_content_type(text: str) -> str:
    """Detect the content type of a tool result."""
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
        return "file"

    if TOK_FORCE_FILE_CODEC and len(text) > 200:
        if _CODE_PATTERNS.search(text) or text.count("\n") > 5:
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
        if line.startswith(("=", "_")):
            in_failure = line.startswith("_ FAILURES") or "FAILED" in line or in_failure
            result.append(line)
            continue
        if in_failure:
            result.append(line)
            continue
        if line.startswith(("collected ", "platform ", "rootdir")):
            result.append(line)

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
        for _i, s in enumerate(shown):
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

    if len(result) >= len(lines):
        return text

    header = f">>> tool:bash|original_lines:{len(lines)}|compressed_lines:{len(result)}"
    return header + "\n" + "\n".join(result)


_SIGNATURE_CONTINUATION_RE = re.compile(r"^[)\],]\s*$|^[)\],]\s*[#]|,\s*$|\S\s*\\$")

_SIGNATURE_OPEN_PAREN_RE = re.compile(r"\(")
_SIGNATURE_CLOSE_PAREN_RE = re.compile(r"\)")


def _is_signature_continuation(prior_unclosed_parens: int, line: str) -> bool:
    if prior_unclosed_parens > 0:
        return True
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith((")", "]", "}")):
        return True
    if stripped.endswith((",", "\\")):
        return True
    if _SIGNATURE_CONTINUATION_RE.match(stripped):
        return True
    return False


_SMALL_FILE_MAX_LINES = 100
_SMALL_FILE_MAX_CHARS = 10000

_SECTION_MAP_RE = re.compile(r"^(class |def |async def )\s*(\w+)")


def _build_section_map(lines: list[str]) -> str:
    """Return a compact 'Name:LN,...' map of top-level scopes for the skeleton header."""
    sections: list[str] = []
    for i, line in enumerate(lines, 1):
        m = _SECTION_MAP_RE.match(line)
        if m:
            sections.append(f"{m.group(2)}:L{i}")
            if len(sections) >= 12:
                break
    return ",".join(sections)


def _is_python_file(text: str, tool_context: dict[str, Any] | None = None) -> bool:
    """Check if the content appears to be Python code."""
    # Check file extension from context
    if tool_context:
        args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
        path = str(
            args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
        )
        if path.endswith(".py") or path.endswith(".pyi"):
            return True
    # Check content heuristics
    python_indicators = [
        r"^\s*def\s+\w+\s*\(",
        r"^\s*class\s+\w+",
        r"^\s*import\s+\w+",
        r"^\s*from\s+\w+\s+import",
        r"^\s*async\s+def\s+\w+",
        r"^\s*@\w+",
        r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]",
    ]
    lines = text.splitlines()[:50]  # Check first 50 lines
    match_count = 0
    for line in lines:
        for pattern in python_indicators:
            if re.match(pattern, line):
                match_count += 1
                if match_count >= 3:
                    return True
    return False


def _extract_python_skeleton(text: str) -> str | None:
    """
    Extract a structural skeleton from Python code using AST.

    Returns the skeleton as a string, or None if parsing fails.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    except ValueError:
        return None

    lines = text.splitlines()
    result: list[str] = []

    def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
        """Extract decorator lines for a node."""
        dec_lines = []
        for dec in node.decorator_list:
            line_num = getattr(dec, "lineno", 0)
            if 1 <= line_num <= len(lines):
                dec_lines.append(lines[line_num - 1].strip())
        return dec_lines

    def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool = False) -> str:
        """Format a function/method signature with type annotations preserved."""
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        func_name = node.name

        # Build argument string
        args_parts: list[str] = []

        def _fmt_arg(arg: ast.arg, default: ast.expr | None = None, prefix: str = "") -> str:
            """Format one argument: [prefix]name[: annotation][ = default]."""
            s = prefix + arg.arg
            if arg.annotation and isinstance(arg.annotation, ast.AST):
                try:
                    s += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            if default is not None:
                try:
                    def_str = ast.unparse(default)
                    s += f" = {def_str}" if len(def_str) <= 30 else " = ..."
                except Exception:
                    s += " = ..."
            return s

        # Handle positional-only args (Python 3.8+)
        if hasattr(node.args, "posonlyargs"):
            for arg in node.args.posonlyargs:
                args_parts.append(_fmt_arg(arg))
            if node.args.posonlyargs:
                args_parts.append("/")

        # Handle regular args (skip 'self' and 'cls' for methods).
        # node.args.defaults aligns to the *tail* of (posonlyargs + args).
        arg_start = 1 if is_method and node.args.args and node.args.args[0].arg in ("self", "cls") else 0
        all_positional = (getattr(node.args, "posonlyargs", []) or []) + node.args.args
        n_defaults = len(node.args.defaults)
        default_start_idx = len(all_positional) - n_defaults
        posonly_count = len(getattr(node.args, "posonlyargs", []) or [])
        for i, arg in enumerate(node.args.args[arg_start:], start=arg_start):
            abs_idx = posonly_count + i
            default = node.args.defaults[abs_idx - default_start_idx] if abs_idx >= default_start_idx else None
            args_parts.append(_fmt_arg(arg, default))

        # Handle varargs (*args) — no default
        if node.args.vararg:
            args_parts.append(_fmt_arg(node.args.vararg, prefix="*"))

        # Handle keyword-only args; kw_defaults is parallel (None means no default)
        for arg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            args_parts.append(_fmt_arg(arg, kw_default))

        # Handle kwargs (**kwargs) — no default
        if node.args.kwarg:
            args_parts.append(_fmt_arg(node.args.kwarg, prefix="**"))

        # Build return annotation
        return_ann = ""
        if node.returns and isinstance(node.returns, ast.AST):
            try:
                ret_str = ast.unparse(node.returns)
                return_ann = f" -> {ret_str}"
            except Exception:
                pass

        sig = f"{prefix}def {func_name}({', '.join(args_parts)}){return_ann}:"
        return sig

    def _extract_body_summary(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        """Extract a summary of the body (return/yield/raise statements, docstring)."""
        body_summary: list[str] = []

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom)):
                try:
                    val_str = ast.unparse(child.value) if child.value else ""
                    if val_str:
                        if len(val_str) > 40:
                            val_str = val_str[:37] + "..."
                        if isinstance(child, ast.Return):
                            body_summary.append(f"return {val_str}")
                        elif isinstance(child, ast.YieldFrom):
                            body_summary.append(f"yield from {val_str}")
                        else:
                            body_summary.append(f"yield {val_str}")
                except Exception:
                    pass
            elif isinstance(child, ast.Raise):
                try:
                    exc_str = ast.unparse(child.exc) if child.exc else ""
                    if exc_str and len(exc_str) <= 40:
                        body_summary.append(f"raise {exc_str}")
                    else:
                        body_summary.append("raise ...")
                except Exception:
                    body_summary.append("raise ...")
            elif isinstance(child, ast.Assign):
                # Check for dataclass-style field assignments
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        # Module-level constants
                        try:
                            val_str = ast.unparse(child.value) if child.value else ""
                            if val_str and len(val_str) <= 30:
                                body_summary.append(f"{target.id} = {val_str}")
                            else:
                                body_summary.append(f"{target.id} = ...")
                        except Exception:
                            body_summary.append(f"{target.id} = ...")

        return " | ".join(body_summary[:3])  # Limit to 3 items

    def _process_class_body(node: ast.ClassDef, indent: str = "  ") -> list[str]:
        """Process class body to extract method signatures and field declarations."""
        class_lines: list[str] = []

        # Process methods and nested classes
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Add decorators
                for dec_line in _get_decorators(child):
                    class_lines.append(f"{indent}{dec_line}")

                # Add method signature
                sig = _format_function_signature(child, is_method=True)
                class_lines.append(f"{indent}{sig}")

                # Add body summary if it has interesting content
                summary = _extract_body_summary(child)
                if summary:
                    class_lines.append(f"{indent}  # {summary}")

            elif isinstance(child, ast.ClassDef):
                # Nested class
                for dec_line in _get_decorators(child):
                    class_lines.append(f"{indent}{dec_line}")
                class_lines.append(f"{indent}class {child.name}:")
                nested_lines = _process_class_body(child, indent + "  ")
                class_lines.extend(nested_lines)

            elif isinstance(child, ast.AnnAssign):
                # Annotated assignments: `name: Type` or `name: Type = default`
                # These are the standard form for dataclass fields and typed class vars.
                if isinstance(child.target, ast.Name):
                    field_name = child.target.id
                    try:
                        ann_str = ast.unparse(child.annotation)
                    except Exception:
                        ann_str = "..."
                    if child.value is not None:
                        try:
                            val_str = ast.unparse(child.value)
                            val_part = f" = {val_str}" if len(val_str) <= 30 else " = ..."
                        except Exception:
                            val_part = " = ..."
                    else:
                        val_part = ""
                    class_lines.append(f"{indent}{field_name}: {ann_str}{val_part}")

            elif isinstance(child, ast.Assign):
                # Untyped field assignments (class variables, dataclass fields)
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        field_name = target.id
                        # Skip numeric constant tables and simple values
                        is_simple_numeric = isinstance(child.value, ast.Constant) and isinstance(
                            child.value.value, (int, float)
                        )
                        if not (is_simple_numeric and field_name.isupper()):
                            # This is likely a field, not a constant
                            try:
                                if isinstance(child.value, ast.Constant):
                                    val = child.value.value
                                    if isinstance(val, str) and len(val) <= 30:
                                        class_lines.append(f"{indent}{field_name} = '{val}'")
                                    elif isinstance(val, (int, float)):
                                        class_lines.append(f"{indent}{field_name} = ...")
                                    else:
                                        class_lines.append(f"{indent}{field_name} = ...")
                                elif isinstance(child.value, ast.Call):
                                    # Likely dataclass field() call
                                    call_str = ast.unparse(child.value)
                                    if len(call_str) <= 40:
                                        class_lines.append(f"{indent}{field_name} = {call_str}")
                                    else:
                                        class_lines.append(f"{indent}{field_name} = field(...)")
                                else:
                                    class_lines.append(f"{indent}{field_name} = ...")
                            except Exception:
                                class_lines.append(f"{indent}{field_name} = ...")

        return class_lines

    # Process module-level nodes
    for node in tree.body:
        if isinstance(node, ast.Import):
            # Import statements
            names = [alias.name for alias in node.names]
            result.append(f"import {', '.join(names)}")

        elif isinstance(node, ast.ImportFrom):
            # From ... import statements
            module = node.module or ""
            names = [alias.name for alias in node.names]
            result.append(f"from {module} import {', '.join(names)}")

        elif isinstance(node, ast.AnnAssign):
            # Module-level annotated assignments: `name: Type` or `name: Type = val`
            if isinstance(node.target, ast.Name):
                field_name = node.target.id
                try:
                    ann_str = ast.unparse(node.annotation)
                except Exception:
                    ann_str = "..."
                if node.value is not None:
                    try:
                        val_str = ast.unparse(node.value)
                        val_part = f" = {val_str}" if len(val_str) <= 30 else " = ..."
                    except Exception:
                        val_part = " = ..."
                else:
                    val_part = ""
                result.append(f"{field_name}: {ann_str}{val_part}")

        elif isinstance(node, ast.Assign):
            # Module-level assignments (constants, module-level variables)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Keep named constants (ALL_CAPS), skip numeric tables
                    if target.id.isupper():
                        try:
                            val_str = ast.unparse(node.value) if node.value else ""
                            # Skip large numeric dicts/lists
                            if isinstance(node.value, (ast.Dict, ast.List, ast.Tuple)):
                                size_hint = "dict" if isinstance(node.value, ast.Dict) else "list/tuple"
                                result.append(f"{target.id} = <{size_hint}>")
                            elif val_str and len(val_str) <= 30:
                                result.append(f"{target.id} = {val_str}")
                            else:
                                result.append(f"{target.id} = ...")
                        except Exception:
                            result.append(f"{target.id} = ...")
                    # Keep dataclass-style field assignments
                    elif isinstance(node.value, ast.Call):
                        func_name = ""
                        if isinstance(node.value.func, ast.Name):
                            func_name = node.value.func.id
                        elif isinstance(node.value.func, ast.Attribute):
                            func_name = node.value.func.attr
                        if func_name in ("field", "Field", "dataclass"):
                            try:
                                call_str = ast.unparse(node.value)
                                if len(call_str) <= 40:
                                    result.append(f"{target.id} = {call_str}")
                                else:
                                    result.append(f"{target.id} = field(...)")
                            except Exception:
                                result.append(f"{target.id} = field(...)")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function definitions
            for dec_line in _get_decorators(node):
                result.append(dec_line)
            sig = _format_function_signature(node, is_method=False)
            result.append(sig)

            # Add body summary for interesting content
            summary = _extract_body_summary(node)
            if summary:
                result.append(f"  # {summary}")

        elif isinstance(node, ast.ClassDef):
            # Class definitions
            for dec_line in _get_decorators(node):
                result.append(dec_line)

            # Build class declaration with bases
            bases: list[str] = []
            for base in node.bases:
                try:
                    base_str = ast.unparse(base)
                    bases.append(base_str)
                except Exception:
                    pass

            if bases:
                result.append(f"class {node.name}({', '.join(bases)}):")
            else:
                result.append(f"class {node.name}:")

            # Process class body
            class_body = _process_class_body(node, indent="  ")
            if class_body:
                result.extend(class_body)
            else:
                result.append("  pass")

    if not result:
        return None

    return "\n".join(result)


def _compress_file_read(text: str, tool_context: dict[str, Any] | None = None, session: Any | None = None) -> str:
    # Small files are never worth skeletonizing — the token savings are negligible
    # but the friction of losing access to the full content is high.
    # Also skip skeletonization for precision reads (offset/limit based) - these are
    # intentional targeted reads and should not be compressed.
    if tool_context:
        args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
        if any(k in args for k in ("offset", "limit", "start", "end")) or args.get("verbatim"):
            return text
        # Zero-heat check: never compress files that haven't been read before
        file_heat = tool_context.get("file_heat") if isinstance(tool_context, dict) else None
        if file_heat:
            path = str(
                args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
            )
            if path:
                norm_path = path.lower().strip()
                heat = file_heat.get(norm_path, 0.0) if isinstance(file_heat, dict) else 0.0
                if heat == 0.0:
                    return text
    if len(text) <= _SMALL_FILE_MAX_CHARS and text.count("\n") + 1 <= _SMALL_FILE_MAX_LINES:
        return text

    # Try AST-based skeleton extraction for Python files
    if _is_python_file(text, tool_context):
        ast_skeleton = _extract_python_skeleton(text)
        if ast_skeleton is not None:
            original_chars = len(text)
            skeleton_lines = ast_skeleton.count("\n") + 1
            # Build a better section map from the AST skeleton
            section_map = _build_section_map(ast_skeleton.splitlines())
            # Track skeleton delivery for edit interception
            if session and hasattr(session, "_skeleton_delivered_paths") and tool_context:
                args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
                path = str(
                    args.get("path")
                    or args.get("file_path")
                    or args.get("AbsolutePath")
                    or args.get("TargetFile")
                    or ""
                )
                if path:
                    norm_path = path.lower().strip()
                    session._skeleton_delivered_paths.add(norm_path)

            header = (
                f">>> tool:file_read|original_chars:{original_chars}|"
                f"skeleton_lines:{skeleton_lines}|retained_skeleton_lines:{skeleton_lines}|ast_skeleton:true|edit_unsafe:true"
                + (f"|sections:{section_map}" if section_map else "")
            )
            return (
                header
                + "\n# [tok optimized] File unchanged — showing structure to save tokens\n"
                + "# Full content: Read path=... offset=1 (adds limit=N for specific section)\n"
                + ast_skeleton
            )

    # Fall back to heuristic skeletonization for non-Python files
    lines = text.splitlines()
    result: list[str] = []
    in_body = False
    in_signature_continuation = False
    unclosed_parens = 0
    last_signature_closed = True
    body_buf: list[str] = []

    signature_re = re.compile(
        r"^(import |from |class |def |async def |[A-Z_][A-Z0-9_]+ =|\s*def |\s*async def |\s*class )"
    )
    key_line_re = re.compile(r"^\s*(return\s+\S[\S\s]*[+\-*/%=<>!&|]|return\s+\w+\.\w+|yield\s+\S|raise\s+\w+Error\()")

    def _flush_body(last_sig_unclosed: bool) -> None:
        nonlocal in_body, body_buf
        if not body_buf:
            return
        key_lines: list[tuple[int, str]] = []
        for j, bl in enumerate(body_buf):
            if key_line_re.match(bl):
                key_lines.append((j, bl))
        if not key_lines:
            if last_sig_unclosed:
                result.append(f"  |> [{len(body_buf)} lines — signature may be incomplete]")
            else:
                result.append(f"  |> [{len(body_buf)} lines]")
        else:
            prev = 0
            for kj, kl in key_lines:
                skipped = kj - prev
                if skipped > 0:
                    if prev == 0 and last_sig_unclosed:
                        result.append(f"  |> [{skipped} lines — signature may be incomplete]")
                    else:
                        result.append(f"  |> [{skipped} lines]")
                result.append(kl)
                prev = kj + 1
            remaining = len(body_buf) - prev
            if remaining > 0:
                result.append(f"  |> [{remaining} lines]")
        body_buf = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_signature_continuation:
                result.append(line)
                continue
            if in_body:
                body_buf.append(line)
            else:
                result.append(line)
            continue

        if signature_re.match(line) or re.match(r"^\s+(def |async def |class )", line):
            if in_body:
                _flush_body(not last_signature_closed)
            in_body = False
            in_signature_continuation = True
            opens = len(_SIGNATURE_OPEN_PAREN_RE.findall(line))
            closes = len(_SIGNATURE_CLOSE_PAREN_RE.findall(line))
            unclosed_parens = max(0, opens - closes)
            last_signature_closed = unclosed_parens == 0
            result.append(line)
            continue

        if _is_signature_continuation(unclosed_parens, line):
            opens = len(_SIGNATURE_OPEN_PAREN_RE.findall(line))
            closes = len(_SIGNATURE_CLOSE_PAREN_RE.findall(line))
            unclosed_parens = max(0, unclosed_parens + opens - closes)
            if opens > 0 or closes > 0:
                in_signature_continuation = True
            if unclosed_parens == 0 and closes > 0:
                in_signature_continuation = False
                last_signature_closed = True
            result.append(line)
            continue

        if in_signature_continuation:
            in_signature_continuation = False
            unclosed_parens = 0
            last_signature_closed = True

        if not in_body:
            in_body = True

        body_buf.append(line)

    if in_body:
        _flush_body(not last_signature_closed)

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
    section_map = _build_section_map(lines)
    # Track skeleton delivery for edit interception
    if session and hasattr(session, "_skeleton_delivered_paths") and tool_context:
        args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
        path = str(
            args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
        )
        if path:
            norm_path = path.lower().strip()
            session._skeleton_delivered_paths.add(norm_path)

    header = (
        f">>> tool:file_read|original_chars:{original_chars}|"
        f"skeleton_lines:{len(result)}|retained_skeleton_lines:{len(trimmed_result)}|edit_unsafe:true"
        + (f"|sections:{section_map}" if section_map else "")
    )
    return (
        header
        + "\n# [tok optimized] File unchanged — showing structure to save tokens\n"
        + "# Full content: Read path=... offset=1 (adds limit=N for specific section)\n"
        + compressed
    )


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
    return header + "\n" + "\n".join(result)


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
    @symbols block with line numbers, e.g.:

        @symbols
          |> L42  import hashlib
          |> L87  class BridgeMemoryState:
          |> L112 def wire_state(self, ...)
          |> L340 def from_tok(cls, ...)

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
    # If already compressed (e.g., skeletonized by _compress_file_read), don't truncate further
    if already_compressed:
        return text

    if len(text) <= int(limit * 1.5):
        return text

    # Don't truncate small multi-line files at default limits - they're high-value discovery targets
    # But still truncate if caller explicitly requests a smaller limit
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
