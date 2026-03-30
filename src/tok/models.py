"""Backward-compatible facade for tok.protocol.models."""

from __future__ import annotations

from .protocol.models import *  # noqa: F403
from .protocol.models import TokNode, build_tok_traceback  # noqa: F401
