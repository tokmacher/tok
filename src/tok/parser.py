"""Backward-compatible facade for tok.protocol.parser."""

from __future__ import annotations

from .protocol.parser import *  # noqa: F403
from .protocol.parser import TokParser  # noqa: F401
