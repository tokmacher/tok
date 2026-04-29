"""File-read codecs (skeletonization) for tool results."""

from __future__ import annotations

import ast
import re
from typing import Any

from tok.runtime.repeat_targets import normalize_path_target

_SIGNATURE_CONTINUATION_RE = re.compile(r"^[)\],]\s*$|^[)\],]\s*[#]|,\s*$|\S\s*\\$")

_SIGNATURE_OPEN_PAREN_RE = re.compile(r"\(")
_SIGNATURE_CLOSE_PAREN_RE = re.compile(r"\)")

_MAX_LITERAL_CHARS: int = 200
_MAX_CONTAINER_ITEMS: int = 30
_MAX_NESTING_DEPTH: int = 2


def _render_literal_value(node: ast.expr, depth: int = 0) -> str | None:
    """Render a simple AST literal node to a Python repr string.

    Returns None when the value is complex (function call, comprehension,
    name reference, etc.) or exceeds size limits.  Never evaluates code.
    Callers should fall back to existing `...` rendering when None is returned.
    """
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return "None"
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, int | float):
            return repr(v)
        if isinstance(v, str):
            rendered = repr(v)
            return rendered if len(rendered) <= _MAX_LITERAL_CHARS else None
        return None

    if depth >= _MAX_NESTING_DEPTH:
        return None

    if isinstance(node, ast.Tuple):
        if len(node.elts) > _MAX_CONTAINER_ITEMS:
            return None
        parts = [_render_literal_value(elt, depth + 1) for elt in node.elts]
        if any(p is None for p in parts):
            return None
        inner = ", ".join(parts)
        rendered = f"({inner},)" if len(node.elts) == 1 else f"({inner})"
        return rendered if len(rendered) <= _MAX_LITERAL_CHARS else None

    if isinstance(node, ast.List):
        if len(node.elts) > _MAX_CONTAINER_ITEMS:
            return None
        parts = [_render_literal_value(elt, depth + 1) for elt in node.elts]
        if any(p is None for p in parts):
            return None
        rendered = "[" + ", ".join(parts) + "]"
        return rendered if len(rendered) <= _MAX_LITERAL_CHARS else None

    if isinstance(node, ast.Set):
        if len(node.elts) > _MAX_CONTAINER_ITEMS:
            return None
        parts = [_render_literal_value(elt, depth + 1) for elt in node.elts]
        if any(p is None for p in parts):
            return None
        rendered = "{" + ", ".join(parts) + "}"
        return rendered if len(rendered) <= _MAX_LITERAL_CHARS else None

    if isinstance(node, ast.Dict):
        if len(node.keys) > _MAX_CONTAINER_ITEMS:
            return None
        pairs: list[str] = []
        for dk, dv in zip(node.keys, node.values, strict=False):
            if dk is None:
                return None
            kr = _render_literal_value(dk, depth + 1)
            vr = _render_literal_value(dv, depth + 1)
            if kr is None or vr is None:
                return None
            pairs.append(f"{kr}: {vr}")
        rendered = "{" + ", ".join(pairs) + "}"
        return rendered if len(rendered) <= _MAX_LITERAL_CHARS else None

    return None


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
    _NON_PYTHON_EXTS = {
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".swift",
        ".kt",
        ".scala",
        ".cs",
    }
    if tool_context:
        args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
        path = str(
            args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
        )
        if path.endswith(".py") or path.endswith(".pyi"):
            return True
        lower_path = path.lower()
        for ext in _NON_PYTHON_EXTS:
            if lower_path.endswith(ext):
                return False
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
            if isinstance(child, ast.Return | ast.Yield | ast.YieldFrom):
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
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
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
                            child.value.value, int | float
                        )
                        if not (is_simple_numeric and field_name.isupper()):
                            # This is likely a field, not a constant
                            try:
                                if isinstance(child.value, ast.Constant):
                                    val = child.value.value
                                    if isinstance(val, str) and len(val) <= 30:
                                        class_lines.append(f"{indent}{field_name} = '{val}'")
                                    elif isinstance(val, int | float):
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
                    lit = _render_literal_value(node.value)
                    if lit is not None:
                        val_part = f" = {lit}"
                    else:
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
                            lit = _render_literal_value(node.value) if node.value else None
                            if lit is not None:
                                result.append(f"{target.id} = {lit}")
                            else:
                                val_str = ast.unparse(node.value) if node.value else ""
                                if val_str and len(val_str) <= 30:
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

        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
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


_OBSERVABILITY_PATH_FRAGMENTS = frozenset(
    {
        "bridge.log",
        "collector.log",
        ".tok/bridge.log",
        ".tok/collector.log",
    }
)


def _is_observability_file(tool_context: dict[str, Any] | None) -> bool:
    if not tool_context:
        return False
    args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
    path = str(args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or "")
    if not path:
        return False
    path_lower = path.lower()
    return any(frag in path_lower for frag in _OBSERVABILITY_PATH_FRAGMENTS)


def _compress_file_read(text: str, tool_context: dict[str, Any] | None = None, session: Any | None = None) -> str:
    _agg = 1.0
    if tool_context and isinstance(tool_context, dict):
        mp = tool_context.get("_model_profile")
        if mp is not None:
            _agg = getattr(mp, "compression_aggressiveness", 1.0)
    elif session is not None:
        mp = getattr(session, "model_profile", None)
        if mp is not None:
            _agg = getattr(mp, "compression_aggressiveness", 1.0)
    if _agg < 0.5:
        return text
    _small_chars = _SMALL_FILE_MAX_CHARS
    _small_lines = _SMALL_FILE_MAX_LINES
    if _agg < 0.8:
        _small_chars = _SMALL_FILE_MAX_CHARS * 3
        _small_lines = _SMALL_FILE_MAX_LINES * 3
    if tool_context:
        args = tool_context.get("args") if isinstance(tool_context.get("args"), dict) else {}
        if any(k in args for k in ("offset", "limit", "start", "end")) or args.get("verbatim"):
            return text
        file_heat = tool_context.get("file_heat") if isinstance(tool_context, dict) else None
        if file_heat is not None:
            path = str(
                args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
            )
            if path:
                norm_path = normalize_path_target(path)
                heat = file_heat.get(norm_path, 0.0) if isinstance(file_heat, dict) else 0.0
                if heat == 0.0:
                    return text
    if len(text) <= _small_chars and text.count("\n") + 1 <= _small_lines:
        return text

    if _is_observability_file(tool_context):
        return text

    # Try AST-based skeleton extraction for Python files
    if _is_python_file(text, tool_context):
        ast_skeleton = _extract_python_skeleton(text)
        if ast_skeleton is not None:
            original_chars = len(text)
            skeleton_lines = ast_skeleton.count("\n") + 1
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
                    norm_path = normalize_path_target(path)
                    session._skeleton_delivered_paths.add(norm_path)

            header = (
                f">>> tool:file_read|original_chars:{original_chars}|"
                f"skeleton_lines:{skeleton_lines}|retained_skeleton_lines:{skeleton_lines}|ast_skeleton:true|"
                "is_skeleton:true|fidelity:summary|lossy:true" + (f"|sections:{section_map}" if section_map else "")
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

    if len("\n".join(result)) >= len(text):
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
            norm_path = normalize_path_target(path)
            session._skeleton_delivered_paths.add(norm_path)

    header = (
        f">>> tool:file_read|original_chars:{original_chars}|"
        f"skeleton_lines:{len(result)}|retained_skeleton_lines:{len(trimmed_result)}|"
        "is_skeleton:true|fidelity:summary|lossy:true" + (f"|sections:{section_map}" if section_map else "")
    )
    return (
        header
        + "\n# [tok optimized] File unchanged — showing structure to save tokens\n"
        + "# Full content: Read path=... offset=1 (adds limit=N for specific section)\n"
        + compressed
    )
