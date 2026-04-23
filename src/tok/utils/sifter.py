"""Code sifter utilities for extracting function signatures and generating hashes."""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any


def extract_args_info(args: ast.arguments) -> list[str]:
    """Extract argument names from AST arguments node."""
    result: list[str] = []
    for arg in args.posonlyargs:
        result.append(arg.arg)
    for arg in args.args:
        result.append(arg.arg)
    for arg in args.kwonlyargs:
        result.append(arg.arg)
    if args.vararg:
        result.append(f"*{args.vararg.arg}")
    if args.kwarg:
        result.append(f"**{args.kwarg.arg}")
    return result


def extract_type_annotation(node: ast.AST) -> str | None:
    """Extract type annotation from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if isinstance(node.slice, ast.Tuple):
                subs = ", ".join(e for e in (extract_type_annotation(e) for e in node.slice.elts) if e)
                return f"{base}[{subs}]"
            if isinstance(node.slice, ast.Name):
                return f"{base}[{node.slice.id}]"
            return base
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    return None


def extract_function_signature(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any]:
    """Extract function signature info from function definition AST node."""
    args = extract_args_info(func_def.args)
    defaults = func_def.args.defaults

    arg_defaults: list[str] = []
    if defaults:
        for i, default in enumerate(defaults):
            default_idx = len(args) - len(defaults) + i
            if default:
                arg_defaults.append(f"{args[default_idx]}={extract_type_annotation(default) or '?'}")

    returns = None
    if func_def.returns:
        returns = extract_type_annotation(func_def.returns)

    return {
        "name": func_def.name,
        "args": args,
        "arg_defaults": arg_defaults,
        "returns": returns,
        "decorators": [d.id for d in func_def.decorator_list if isinstance(d, ast.Name)],
    }


def extract_class_info(class_def: ast.ClassDef) -> dict[str, Any]:
    """Extract class info from class definition AST node."""
    bases = []
    for base in class_def.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(f"{extract_type_annotation(base.value)}.{base.attr}")

    return {
        "name": class_def.name,
        "bases": bases,
        "decorators": [d.id for d in class_def.decorator_list if isinstance(d, ast.Name)],
    }


def _minify_python(source_code: str) -> str:
    """Strip blank lines and # comments, keep docstrings."""
    lines: list[str] = []
    for line in source_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _preserve_python(source_code: str) -> str:
    """Preserve all content including blank lines and comments."""
    return source_code


def _get_source_segment(source: str, node: ast.AST) -> str:
    """Extract source by line numbers, preserving decorators."""
    lines = source.split("\n")
    start = node.lineno - 1  # type: ignore[attr-defined]
    end = node.end_lineno  # type: ignore[attr-defined]

    # Include decorator lines before the node
    while start > 0 and lines[start - 1].strip().startswith("@"):
        start -= 1

    return "\n".join(lines[start:end])


class DirectoryWalker:
    """Walk directory tree to find Python files, excluding patterns."""

    def __init__(self, exclude_patterns: list[str] | None = None) -> None:
        """Initialize walker with exclude patterns."""
        self.exclude_patterns = exclude_patterns or [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            "node_modules",
            "*.pyc",
            ".DS_Store",
        ]

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded based on patterns."""
        name = path.name
        parts = path.parts
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in parts:
                return True
        return False

    def walk(self, root_path: str) -> list[Path]:
        """Walk directory and return list of Python files."""
        root = Path(root_path)
        py_files: list[Path] = []

        for path in root.rglob("*.py"):
            if self.should_exclude(path):
                continue
            # Skip test files but keep __init__.py
            if path.name.startswith("test_"):
                continue
            # Include __init__.py
            if path.name == "__init__.py":
                py_files.append(path)
                continue
            if any(part.startswith("test_") for part in path.parts):
                continue
            py_files.append(path)

        return sorted(py_files)


class Sifter:
    """Sift Python code into Tok format with pointer references."""

    _pointer_counter = 0

    def __init__(self) -> None:
        """Initialize sifter with directory walker."""
        self.walker = DirectoryWalker()
        self.corpus: dict[str, str] = {}

    @classmethod
    def _get_next_pointer(cls) -> str:
        n = cls._pointer_counter
        cls._pointer_counter += 1
        result = ""
        while n >= 0:
            result = chr((n % 26) + 65) + result
            n = (n // 26) - 1
        return result

    @classmethod
    def reset_pointers(cls) -> None:
        """Reset pointer sequence so tests and deterministic runs stay stable."""
        cls._pointer_counter = 0

    def _add_method_entries(
        self,
        node: ast.ClassDef,
        pointer_id: str,
        naked: bool,
        lines: list[str],
    ) -> None:
        for item in node.body:
            if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if item.name in ("__init__", "__repr__", "__str__"):
                continue
            if item.name.startswith("_"):
                continue

            func_info = extract_function_signature(item)
            args = [a for a in func_info["args"] if a not in ("self", "cls")]

            args_str = args[0] if naked and len(args) > 1 else ", ".join(args)

            if naked:
                if args_str:
                    lines.append(f"  *{pointer_id} {item.name}({args_str})")
                else:
                    lines.append(f"  *{pointer_id} {item.name}()")
            elif args_str:
                lines.append(f"  @func {item.name}({args_str}) ref:*{pointer_id}")
            else:
                lines.append(f"  @func {item.name}() ref:*{pointer_id}")

    def _add_func_entry(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        normalize: Callable[[str], str],
        naked: bool,
        lines: list[str],
    ) -> None:
        if node.name.startswith("_"):
            return

        func_info = extract_function_signature(node)
        node_source = _get_source_segment(source, node) or ""
        pointer_id = self._get_next_pointer()
        self.corpus[pointer_id] = normalize(node_source)

        args = [a for a in func_info["args"] if a not in ("self", "cls")]

        args_str = args[0] if naked and len(args) > 1 else ", ".join(args)

        if naked:
            if args_str:
                lines.append(f"  *{pointer_id} {node.name}({args_str})")
            else:
                lines.append(f"  *{pointer_id} {node.name}()")
        elif args_str:
            lines.append(f"  @func {node.name}({args_str}) ref:*{pointer_id}")
        else:
            lines.append(f"  @func {node.name}() ref:*{pointer_id}")

    @staticmethod
    def _collect_deps(tree: ast.Module, source: str) -> list[str]:
        deps_lines: list[str] = []
        seen_import = False
        seen_assign = False
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                segment = _get_source_segment(source, node)
                if segment:
                    deps_lines.append(segment)
                seen_import = True
            elif isinstance(node, ast.Assign):
                segment = _get_source_segment(source, node)
                if segment:
                    if seen_import and not seen_assign and deps_lines:
                        deps_lines.append("")
                    deps_lines.append(segment)
                seen_assign = True
        return deps_lines

    @staticmethod
    def _resolve_module_name(filepath: Path) -> str:
        try:
            rel_parts = filepath.with_suffix("").parts
            if "tok" in rel_parts:
                idx = rel_parts.index("tok")
                return ".".join(rel_parts[idx:])
        except Exception:
            pass
        return filepath.stem

    def _sift_class_node(
        self,
        node: ast.ClassDef,
        source: str,
        normalize: Callable[[str], str],
        naked: bool,
        lines: list[str],
    ) -> None:
        class_info = extract_class_info(node)
        node_source = _get_source_segment(source, node) or ""
        pointer_id = self._get_next_pointer()
        self.corpus[pointer_id] = normalize(node_source)

        if naked:
            lines.append(f"  *{pointer_id} {class_info['name']}")
        else:
            bases_str = f" bases:{','.join(class_info['bases'])}" if class_info["bases"] else ""
            lines.append(f"  @class {class_info['name']} ref:*{pointer_id}{bases_str}")

        self._add_method_entries(node, pointer_id, naked, lines)

    def sift_file(self, filepath: str | Path, naked: bool = False, minify: bool = True) -> list[str]:
        """Sift a single Python file into Tok format."""
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        module_name = self._resolve_module_name(filepath)
        lines: list[str] = [f"@{module_name}"]

        deps_lines = self._collect_deps(tree, source)
        if deps_lines:
            deps_pointer = self._get_next_pointer()
            self.corpus[deps_pointer] = "\n".join(deps_lines)
            if naked:
                lines.append(f"  *{deps_pointer} deps")
            else:
                lines.append(f"  @deps ref:*{deps_pointer}")

        normalize = _minify_python if minify else _preserve_python
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._sift_class_node(node, source, normalize, naked, lines)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self._add_func_entry(node, source, normalize, naked, lines)

        return lines

    def sift_directory(self, root_path: str, naked: bool = False, minify: bool = True) -> list[str]:
        """Sift a directory of Python files into Tok format."""
        py_files = self.walker.walk(root_path)

        if naked:
            lines = [f"@{Path(root_path).name}"]
        else:
            lines = [
                "@meta v:1.8 sifter:recursive",
                '@protocol directive:"ALWAYS_INVERT" context:"Sovereign Inversion. Prefer node body for code payloads."',
                f"@repo {Path(root_path).name}",
                "",
            ]

        for filepath in py_files:
            file_lines = self.sift_file(filepath, naked=naked, minify=minify)
            if len(file_lines) <= 1:
                continue
            lines.extend(file_lines)
            lines.append("")

        lines.extend(["@corpus", ""])

        for ref_hash, body in self.corpus.items():
            lines.append(f"  @chunk ref:*{ref_hash}")
            lines.append(f"    |#{ref_hash}>")
            for body_line in body.splitlines():
                lines.append(f"    {body_line}")
            lines.append(f"    |#{ref_hash}")
            lines.append("")

        return lines

    @staticmethod
    def from_dir(
        path: str,
        exclude: list[str] | None = None,
        naked: bool = False,
        minify: bool = True,
    ) -> str:
        sifter = Sifter()
        if exclude:
            sifter.walker.exclude_patterns.extend(exclude)
        lines = sifter.sift_directory(path, naked=naked, minify=minify)
        return "\n".join(lines)

    @staticmethod
    def from_file(filepath: str, naked: bool = False, minify: bool = True) -> dict[str, Any]:
        sifter = Sifter()
        path = Path(filepath)
        skeleton_lines = sifter.sift_file(path, naked=naked, minify=minify)
        return {"skeleton": "\n".join(skeleton_lines), "corpus": sifter.corpus}

    @staticmethod
    def _dedent_content(content: str, base_indent: int) -> str:
        if base_indent <= 0:
            return content
        prefix = " " * base_indent
        dedented: list[str] = []
        for line in content.split("\n"):
            dedented.append(line[base_indent:] if line.startswith(prefix) else line)
        return "\n".join(dedented)

    @staticmethod
    def _parse_corpus_chunks(tok_string: str) -> dict[str, str]:
        import re

        chunks: dict[str, str] = {}
        current_ref: str | None = None
        in_chunk = False
        chunk_lines: list[str] = []
        base_indent = 0

        for line in tok_string.split("\n"):
            stripped = line.strip()
            if stripped.startswith("@chunk") and "ref:" in stripped:
                match = re.search(r"ref:\*(\w+)", stripped)
                if match:
                    current_ref = match.group(1)
                    in_chunk = True
                    chunk_lines = []
            elif in_chunk and re.match(r"^\|#[A-Z]+\>$", stripped):
                base_indent = len(line) - len(line.lstrip())
            elif in_chunk and stripped == f"|#{current_ref}":
                content = Sifter._dedent_content("\n".join(chunk_lines), base_indent)
                if current_ref is not None:
                    chunks[current_ref] = content
                in_chunk = False
                current_ref = None
                base_indent = 0
            elif in_chunk:
                chunk_lines.append(line)

        return chunks

    @staticmethod
    def _extract_ref(stripped: str) -> str | None:
        import re

        match = re.search(r"ref:\*(\w+)", stripped)
        return match.group(1) if match else None

    @staticmethod
    def _process_member_line(
        stripped: str,
        current_module: str | None,
        module_members: dict[str, dict[str, Any]],
        skip_node_types: tuple[str, ...],
        skip_prefixes: tuple[str, ...],
    ) -> str | None:
        if (
            stripped.startswith("@")
            and not stripped.startswith("@chunk")
            and not any(stripped.startswith(p) for p in skip_prefixes)
        ):
            parts = stripped.split()
            if parts:
                node_type = parts[0][1:]
                if node_type not in skip_node_types:
                    current_module = node_type
                    module_members[current_module] = {
                        "deps": None,
                        "items": [],
                    }

        if current_module and "@deps" in stripped:
            ref = Sifter._extract_ref(stripped)
            if ref:
                module_members[current_module]["deps"] = ref
        elif current_module and ("@class" in stripped or "@func" in stripped):
            ref = Sifter._extract_ref(stripped)
            if ref:
                module_members[current_module]["items"].append(ref)

        return current_module

    @staticmethod
    def _parse_module_members(
        tok_string: str,
    ) -> dict[str, dict[str, Any]]:
        skip_node_types = (
            "func",
            "class",
            "chunk",
            "meta",
            "repo",
            "deps",
        )
        skip_prefixes = ("@meta", "@repo")

        current_module: str | None = None
        module_members: dict[str, dict[str, Any]] = {}
        in_corpus = False

        for line in tok_string.split("\n"):
            stripped = line.strip()

            if stripped == "@corpus":
                in_corpus = True
            if in_corpus:
                continue

            current_module = Sifter._process_member_line(
                stripped,
                current_module,
                module_members,
                skip_node_types,
                skip_prefixes,
            )

        return module_members

    @staticmethod
    def _write_module(
        module_name: str,
        data: dict[str, Any],
        chunks: dict[str, str],
        out_dir: Path,
    ) -> None:
        if not data["items"] and not data["deps"]:
            return

        file_name = module_name.rsplit(".", maxsplit=1)[-1]
        file_path = out_dir / f"{file_name}.py"

        content_parts: list[str] = []
        if data["deps"] and data["deps"] in chunks:
            assert isinstance(data["deps"], str)
            content_parts.append(chunks[data["deps"]])

        unique_refs: list[str] = []
        seen: set[str] = set()
        for ref in data["items"]:
            if ref not in seen:
                seen.add(ref)
                unique_refs.append(ref)

        for ref in unique_refs:
            if ref in chunks:
                assert isinstance(ref, str)
                content_parts.append(chunks[ref])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(content_parts) + "\n")
