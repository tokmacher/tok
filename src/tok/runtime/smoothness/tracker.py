"""Smoothness event tracker for collecting events per turn.

This module provides the SmoothnessTracker class that collects smoothness
events during a turn and produces reports.
"""

from __future__ import annotations

import uuid
from typing import Any

from .models import (
    SmoothnessEvent,
    SmoothnessEventType,
    TaskSmoothnessReport,
    TurnSmoothnessReport,
)
from .policy import choose_tok_mode
from .scoring import PENALTIES, score_task, score_turn


class SmoothnessTracker:
    """Tracks smoothness events within a turn and across a task.

    Usage:
        tracker = SmoothnessTracker()
        tracker.start_turn("turn_1", "task_abc")
        tracker.record(SmoothnessEventType.STREAM_READ_ERROR, {"path": "foo.py"})
        report = tracker.finish_turn()
    """

    def __init__(self) -> None:
        self._current_turn_id: str | None = None
        self._current_task_id: str | None = None
        self._current_events: list[SmoothnessEvent] = []
        self._turn_reports: list[TurnSmoothnessReport] = []
        self._task_reports: dict[str, TaskSmoothnessReport] = {}

    def start_turn(
        self, turn_id: str | None = None, task_id: str | None = None
    ) -> None:
        """Start a new turn, optionally specifying IDs.

        If turn_id or task_id are not provided, UUIDs will be generated.
        """
        if turn_id is None:
            turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

        self._current_turn_id = turn_id
        self._current_task_id = task_id
        self._current_events = []

    def record(
        self,
        event_type: SmoothnessEventType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a smoothness event for the current turn.

        Args:
            event_type: Type of event that occurred
            metadata: Optional additional context about the event

        Raises:
            RuntimeError: If called before start_turn()
        """
        if self._current_turn_id is None or self._current_task_id is None:
            raise RuntimeError("Must call start_turn() before record()")

        penalty = PENALTIES.get(event_type, 0)

        event = SmoothnessEvent(
            event_type=event_type,
            turn_id=self._current_turn_id,
            task_id=self._current_task_id,
            penalty=penalty,
            metadata=metadata or {},
        )

        self._current_events.append(event)

    def finish_turn(self) -> TurnSmoothnessReport:
        """Finish the current turn and compute the smoothness report.

        Returns:
            TurnSmoothnessReport for the completed turn

        Raises:
            RuntimeError: If called before start_turn()
        """
        if self._current_turn_id is None or self._current_task_id is None:
            raise RuntimeError("Must call start_turn() before finish_turn()")

        report = score_turn(
            turn_id=self._current_turn_id,
            task_id=self._current_task_id,
            events=self._current_events,
        )

        # Use policy module to choose the mode
        task_report = self._task_reports.get(self._current_task_id)
        mode = choose_tok_mode(report, task_report)
        report.mode = mode

        self._turn_reports.append(report)

        task_id = self._current_task_id
        task_reports = [r for r in self._turn_reports if r.task_id == task_id]
        self._task_reports[task_id] = score_task(task_id, task_reports)

        self._current_turn_id = None
        self._current_events = []

        return report

    def current_task_report(self) -> TaskSmoothnessReport | None:
        """Get the current task report if available.

        Returns:
            TaskSmoothnessReport for the current task, or None if no turns completed
        """
        if self._current_task_id is None:
            return None

        task_reports = [
            r for r in self._turn_reports if r.task_id == self._current_task_id
        ]
        if not task_reports:
            return None

        return score_task(self._current_task_id, task_reports)

    def get_task_report(self, task_id: str) -> TaskSmoothnessReport | None:
        """Get a specific task report by ID.

        Args:
            task_id: Task identifier

        Returns:
            TaskSmoothnessReport if task exists, None otherwise
        """
        return self._task_reports.get(task_id)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._current_turn_id = None
        self._current_task_id = None
        self._current_events = []
        self._turn_reports = []
        self._task_reports = {}

    @property
    def current_turn_id(self) -> str | None:
        """Current turn ID if a turn is active."""
        return self._current_turn_id

    @property
    def current_task_id(self) -> str | None:
        """Current task ID if a turn is active."""
        return self._current_task_id

    @property
    def turn_count(self) -> int:
        """Total number of completed turns."""
        return len(self._turn_reports)
