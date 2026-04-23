"""
Smoothness scoring and control layer for Tok.

This module provides a passive scoring system that tracks interaction quality
without changing compression behavior in Phase 1.
"""

from .models import (
    SmoothnessEvent,
    SmoothnessEventType,
    TaskSmoothnessReport,
    TokMode,
    TurnSmoothnessReport,
)
from .policy import choose_tok_mode
from .scoring import score_task, score_turn
from .tracker import SmoothnessTracker

__all__ = [
    "SmoothnessEvent",
    "SmoothnessEventType",
    "SmoothnessTracker",
    "TaskSmoothnessReport",
    "TokMode",
    "TurnSmoothnessReport",
    "choose_tok_mode",
    "score_task",
    "score_turn",
]
