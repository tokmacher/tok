from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FidelityState:
    overrides: dict[str, int] = field(default_factory=dict)
    file_reads_by_turn: dict[str, int] = field(default_factory=dict)
    last_elevated_path: str = ""
    tool_required_latch_streak: int = 0

    def reset(self) -> None:
        self.overrides.clear()
        self.file_reads_by_turn.clear()
        self.last_elevated_path = ""
        self.tool_required_latch_streak = 0
