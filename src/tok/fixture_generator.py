"""Backward-compatible shim for fixture generation helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    return getattr(
        import_module(".testing.fixture_generator", __package__), name
    )
