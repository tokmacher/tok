"""Request policy state extracted from RuntimeSession."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RequestPolicyState:
    """Tracks sticky-turn counters and effective mode for request policy."""

    tool_mode_sticky_turns: int = 0
    stream_recovery_watch_turns: int = 0
    tool_recovery_watch_turns: int = 0
    last_effective_tool_compatible: bool = False
    drift_detected_previous_turn: bool = False

    def reset(self) -> None:
        self.tool_mode_sticky_turns = 0
        self.stream_recovery_watch_turns = 0
        self.tool_recovery_watch_turns = 0
        self.last_effective_tool_compatible = False
        self.drift_detected_previous_turn = False
