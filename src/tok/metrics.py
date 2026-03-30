"""Backward-compatible shim for metric helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    return getattr(import_module(".utils.metrics", __package__), name)
