"""Agent exploration utilities for Tok - lightweight code introspection."""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("tok.explorer")

from .utils.sifter import Sifter


def get_file_overview(filepath: str) -> dict[str, Any]:
    """Return structured overview of a Python file.

    Returns:
        dict with keys: path, line_count, classes, functions, is_large
    """
    path = Path(filepath)

    # Validate file path and existence
    if not path.exists():
        return {"error": f"File not found: {filepath}"}

    # Security check: ensure path is within reasonable bounds
    try:
        path.resolve().relative_to(Path.cwd())
    except ValueError:
        # Path is outside current directory - allow but log warning
        logger.warning(
            "Accessing file outside current directory: %s", filepath
        )

    # Ensure it's a file (not directory)
    if not path.is_file():
        return {"error": f"Path is not a file: {filepath}"}

    # Check if it's a Python file
    if path.suffix != ".py":
        return {"error": f"Not a Python file: {filepath}"}

    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        return {"error": str(e)}

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                m.name
                for m in node.body
                if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods,
                    "bases": [
                        b.id if isinstance(b, ast.Name) else None
                        for b in node.bases
                    ],
                }
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [a.arg for a in node.args.args if a.arg != "self"],
                }
            )

    line_count = len(source.splitlines())

    return {
        "path": str(path),
        "line_count": line_count,
        "classes": classes,
        "functions": functions,
        "is_large": line_count > 500,
    }


def explore_file(filepath: str, mode: str = "overview") -> str:
    """Explore a file and return Tok-formatted output.

    Args:
        filepath: Path to Python file
        mode: "overview" for summary, "skeleton" for full structure

    Returns:
        Tok-formatted string
    """
    path = Path(filepath)
    if not path.exists():
        return f"@error File not found: {filepath}"

    if mode == "skeleton":
        result: dict[str, Any] = Sifter.from_file(
            filepath, naked=False, minify=True
        )
        return cast(str, result["skeleton"])

    # overview mode
    overview: dict[str, Any] = get_file_overview(filepath)
    if "error" in overview:
        error_msg = overview["error"]
        return f"@error {error_msg}" if error_msg else "@error Unknown error"

    lines = [
        f"@file {path.name}",
        f"  lines: {overview['line_count']}",
        f"  large: {overview['is_large']}",
    ]

    if overview.get("classes"):
        lines.append("  @class")
        for cls in overview["classes"]:
            methods = ", ".join(cls["methods"]) if cls["methods"] else ""
            lines.append(
                f"    {cls['name']} l:{cls['line']}{' ' + methods if methods else ''}"
            )

    if overview.get("functions"):
        lines.append("  @func")
        for func in overview["functions"]:
            args = ", ".join(func["args"]) if func["args"] else ""
            lines.append(f"    {func['name']}({args}) l:{func['line']}")

    return "\n".join(lines)


def list_large_files(root: str = "src/tok") -> list[dict[str, Any]]:
    """Find all Python files > 500 lines in a directory tree.

    Args:
        root: Root directory to search

    Returns:
        List of dicts with file info
    """
    root_path = Path(root)

    # Validate root path
    if not root_path.exists():
        logger.warning("Root path does not exist: %s", root)
        return []

    if not root_path.is_dir():
        # If root is not a directory, try its parent
        if root_path.parent.exists() and root_path.parent.is_dir():
            root = str(root_path.parent)
            root_path = root_path.parent
        else:
            logger.warning("Invalid root path: %s", root)
            return []

    large_files = []

    for root_dir, dirs, files in os.walk(root):
        # Security: ensure we're not walking outside intended bounds
        try:
            Path(root_dir).resolve().relative_to(Path(root).resolve())
        except ValueError:
            # Skip directories outside the root
            dirs[:] = []
            continue

        # Skip common non-code directories
        dirs[:] = [
            d
            for d in dirs
            if d not in ("__pycache__", ".git", ".venv", "venv", "env")
        ]

        for fname in files:
            if not fname.endswith(".py"):
                continue

            fpath = os.path.join(root_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    line_count = len(f.readlines())
            except Exception:
                continue

            if line_count > 500:
                # Get quick overview
                overview = get_file_overview(fpath)
                large_files.append(
                    {
                        "path": fpath,
                        "line_count": line_count,
                        "class_count": len(overview.get("classes", [])),
                        "function_count": len(overview.get("functions", [])),
                    }
                )

    return sorted(large_files, key=lambda x: x["line_count"], reverse=True)


def explore_module(module_path: str, mode: str = "overview") -> str:
    """Explore a module/package and return Tok-formatted overview.

    Args:
        module_path: Path to module directory or file
        mode: "overview" for summary, "skeleton" for full structure

    Returns:
        Tok-formatted string
    """
    path = Path(module_path)

    if not path.exists():
        return f"@error Path not found: {module_path}"

    if path.is_file() and path.suffix == ".py":
        return explore_file(str(path), mode)

    # Directory/package
    if path.is_dir() and not (path / "__init__.py").exists():
        # Not a package, find py files
        py_files = list(path.glob("*.py"))
        if not py_files:
            return f"@error No Python files in: {module_path}"

        lines = [f"@module {path.name}", ""]
        for pf in sorted(py_files)[:10]:  # Limit to 10 files
            overview = get_file_overview(str(pf))
            lines.append(f"  @file {pf.name}")
            lines.append(f"    lines: {overview.get('line_count', '?')}")
            lines.append(f"    funcs: {len(overview.get('functions', []))}")
            lines.append(f"    classes: {len(overview.get('classes', []))}")
            lines.append("")

        return "\n".join(lines)

    # It's a package
    if mode == "skeleton":
        return Sifter.from_dir(str(path), naked=False, minify=True)

    # Overview mode
    init_file = path / "__init__.py"
    overview = (
        get_file_overview(str(init_file))
        if init_file.exists()
        else {"classes": [], "functions": []}
    )

    lines = [
        f"@module {path.name}",
        "  @file __init__.py",
    ]

    # Find all py files in package
    for pf in sorted(path.glob("*.py")):
        if pf.name.startswith("_"):
            continue
        overview = get_file_overview(str(pf))
        lines.append(f"  @file {pf.name}")
        lines.append(f"    lines: {overview.get('line_count', '?')}")
        lines.append(f"    funcs: {len(overview.get('functions', []))}")
        lines.append(f"    classes: {len(overview.get('classes', []))}")

    return "\n".join(lines)


__all__ = [
    "get_file_overview",
    "explore_file",
    "explore_module",
    "list_large_files",
]
