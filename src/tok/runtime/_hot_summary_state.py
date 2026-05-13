from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HotSummaryState:
    recent_repeat_target_events: list[Any] = field(default_factory=list)
    records: dict[str, Any] = field(default_factory=dict)
    hints_loaded_from_disk: int = 0

    def reset(self, *, save_before_clear: bool = False, session: Any = None) -> None:
        if save_before_clear and session is not None:
            from .core import save_hot_summaries

            save_hot_summaries(session)
        self.recent_repeat_target_events.clear()
        self.records.clear()
        self.hints_loaded_from_disk = 0
