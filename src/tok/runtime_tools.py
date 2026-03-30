"""Backward-compatible facade for tok.runtime.tools."""

from __future__ import annotations

from .runtime.tools import *  # noqa: F403
from .runtime.tools import (
    RuntimeToolExecutor,  # noqa: F401
    execute_normalized_tool,  # noqa: F401
)
