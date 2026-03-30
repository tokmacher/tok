"""Backward-compatible facade for tok.protocol.encoder."""

from __future__ import annotations

from .protocol.encoder import *  # noqa: F403
from .protocol.encoder import TokEncoder  # noqa: F401
