"""Data models for smoothness scoring.

This module defines the core types used throughout the smoothness layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TokMode(str, Enum):
    """Tok compression mode based on smoothness score."""

    FULL_TOK = "full_tok"
    GUARDED_TOK = "guarded_tok"
    SMOOTH_MODE = "smooth_mode"
    LOSSLESS_TASK_MODE = "lossless_task_mode"


class SmoothnessEventType(str, Enum):
    """Event types that impact smoothness score."""

    STREAM_READ_ERROR = "stream_read_error"
    EMPTY_STREAM_SUCCESS = "empty_stream_success"
    STREAM_RECOVERY_STARTED = "stream_recovery_started"
    STREAM_RECOVERY_SUCCEEDED = "stream_recovery_succeeded"
    STREAM_RECOVERY_LOOP_BREAKER = "stream_recovery_loop_breaker"
    UPSTREAM_400_AFTER_PREPARED_PAYLOAD = "upstream_400_after_prepared_payload"
    THINKING_BLOCK_MUTATION = "thinking_block_mutation"
    MESSAGES_CHANGED_OPEN_TOOL_LOOP = "messages_changed_open_tool_loop"
    HISTORY_WINNOWING_ACTIVE_LOOP = "history_winnowing_active_loop"
    SEMANTIC_DEDUP_ACTIVE_FILE = "semantic_dedup_active_file"
    PROMPT_OPTIMIZATION_ACTIVE_TASK = "prompt_optimization_active_task"
    REPEATED_ACTIVE_FILE_READ = "repeated_active_file_read"
    REPEATED_SEARCH_SAME_TARGET = "repeated_search_same_target"
    USER_INTERRUPT_REDIRECTION = "user_interrupt_redirection"
    DIRECT_ACTION_AFTER_FIRST_READ = "direct_action_after_first_read"


@dataclass
class SmoothnessEvent:
    """A single smoothness-impacting event during a turn."""

    event_type: SmoothnessEventType
    turn_id: str
    task_id: str
    penalty: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnSmoothnessReport:
    """Smoothness report for a single turn."""

    turn_id: str
    task_id: str
    score: int
    labour_index: int
    mode: TokMode
    events: list[SmoothnessEvent] = field(default_factory=list)


@dataclass
class TaskSmoothnessReport:
    """Aggregated smoothness report across multiple turns in a task."""

    task_id: str
    average_turn_score: float
    worst_turn_score: int
    task_score: int
    labour_index: int
    event_counts: dict[str, int] = field(default_factory=dict)
    turn_count: int = 0
