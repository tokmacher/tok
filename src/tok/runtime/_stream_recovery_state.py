"""Streaming recovery state extracted from RuntimeSession."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamingRecoveryState:
    """Tracks streaming-error recovery budget and cooldown."""

    reacquisition_budget: int = 0
    history_floor_budget: int = 0
    tool_use_only_signature: str = ""
    tool_use_only_repeat_count: int = 0
    cooldown_remaining: int = 0
    cooldown_suppressed: bool = False
    read_error_consecutive_count: int = 0
    read_error_last_stage: str = ""

    def reset(self) -> None:
        self.reacquisition_budget = 0
        self.history_floor_budget = 0
        self.tool_use_only_signature = ""
        self.tool_use_only_repeat_count = 0
        self.cooldown_remaining = 0
        self.cooldown_suppressed = False
        self.read_error_consecutive_count = 0
        self.read_error_last_stage = ""
