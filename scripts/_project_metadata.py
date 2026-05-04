"""Helpers for reading release-critical project metadata."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_project_metadata(pyproject_path: Path = REPO_ROOT / "pyproject.toml") -> dict[str, str]:
    """Read the project metadata fields mirrored into release artifacts."""
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    return {
        "name": project["name"],
        "version": project["version"],
    }
