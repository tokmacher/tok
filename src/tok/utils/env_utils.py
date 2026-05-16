"""Environment variable parsing utilities.

Keep this module dependency-free to avoid import cycles.
"""

from __future__ import annotations

import os
from typing import Final

_FALSEY: Final[set[str]] = {"0", "false", "off", "no"}


def env_int(name: str, fallback: int, *, legacy_name: str | None = None) -> int:
    """Parse an integer environment variable.

    If legacy_name is provided, it is checked first when name is unset.
    """
    raw = os.getenv(name)
    if raw is None and legacy_name:
        raw = os.getenv(legacy_name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def env_int_or_default(name: str, default: int) -> int:
    """Parse an integer environment variable or return default."""
    return env_int(name, default)


def env_bool(name: str, default: bool = False, *, legacy_name: str | None = None) -> bool:
    raw = os.getenv(name)
    if raw is None and legacy_name:
        raw = os.getenv(legacy_name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in _FALSEY


__all__ = ["env_bool", "env_int", "env_int_or_default"]
