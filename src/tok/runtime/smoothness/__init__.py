"""
Smoothness scoring and control layer for Tok.

This module provides interaction quality scoring that actively controls
compression behavior. The resulting TokMode gates history compression,
winnowing, and raw file delivery across multiple code paths.
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
