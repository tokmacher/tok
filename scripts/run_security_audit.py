#!/usr/bin/env python3

"""Run pip-audit against dependencies exported from uv.lock."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def export_requirements() -> str:
    """Render requirements via uv so markers/platforms are resolved correctly."""
    completed = subprocess.run(
        ["uv", "export", "--format", "requirements-txt", "--no-hashes"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)
    filtered_lines = [
        line
        for line in completed.stdout.splitlines()
        if not line.startswith("-e ")
    ]
    return "\n".join(filtered_lines) + "\n"


def audit_command() -> list[str]:
    """Resolve the most reliable way to invoke pip-audit."""
    return ["uv", "run", "--with", "pip-audit", "pip-audit"]


def main() -> int:
    lock_path = Path("uv.lock")
    if not lock_path.exists():
        print("uv.lock not found", file=sys.stderr)
        return 1

    requirements_text = export_requirements()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".requirements.txt", delete=False
    ) as temp_file:
        temp_file.write(requirements_text)
        temp_path = Path(temp_file.name)

    cmd = [*audit_command(), "--strict", "-r", str(temp_path)]
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
