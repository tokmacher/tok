"""Backward-compatible shim for shell integration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    return getattr(
        import_module(".utils.shell_integration", __package__), name
    )
