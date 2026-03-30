"""Backward-compatible facade for tok.analysis.prompt_analyzer."""

from __future__ import annotations

from .analysis.prompt_analyzer import *  # noqa: F403
from .analysis.prompt_analyzer import count_tokens  # noqa: F401
