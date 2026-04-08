#!/usr/bin/env python3
"""Fail fast when tracked repo clutter slips back into the tree."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

ALLOWED_TOP_LEVEL_FILES = frozenset(
    {
        ".gitignore",
        ".pre-commit-config.yaml",
        ".python-version",
        "CHANGELOG.md",
        "CLAUDE.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "DEVELOPMENT.md",
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "gate-config.json",
        "pyproject.toml",
        "roadmap.md",
        "uv.lock",
    }
)

FORBIDDEN_TRACKED_ROOT_ARTIFACTS = frozenset(
    {
        "execution.log",
        "memory.tok",
        "savings.tok",
        "stats.tok",
        "telemetry.db",
        "todo.tok",
    }
)


def load_tracked_files(repo_root: Path | None = None) -> list[str]:
    cwd = repo_root or Path.cwd()
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=False,
    )
    tracked_files: list[str] = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        path = item.decode("utf-8")
        if not (cwd / path).exists():
            continue
        tracked_files.append(path)
    return tracked_files


def find_violations(tracked_files: Iterable[str]) -> list[str]:
    violations: list[str] = []
    for path in tracked_files:
        if path.endswith(".bak"):
            violations.append(f"tracked backup file: {path}")
        if path.endswith(".pyc") or path.startswith("__pycache__/") or "/__pycache__/" in path:
            violations.append(f"tracked Python cache artifact: {path}")
        if path.startswith("dist/"):
            violations.append(f"tracked build artifact under dist/: {path}")
        if path.startswith("tmp/"):
            violations.append(f"tracked temporary artifact under tmp/: {path}")
        if path in FORBIDDEN_TRACKED_ROOT_ARTIFACTS:
            violations.append(f"tracked runtime artifact in repo root: {path}")
        if "/" not in path and path not in ALLOWED_TOP_LEVEL_FILES:
            violations.append(f"unexpected tracked top-level file: {path}")
    return violations


def main() -> int:
    violations = find_violations(load_tracked_files())
    if not violations:
        return 0

    for violation in violations:
        print(violation)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
