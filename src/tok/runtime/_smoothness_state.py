from __future__ import annotations

from dataclasses import dataclass, field

from .smoothness.models import TokMode


@dataclass
class SmoothnessState:
    latest_turn_score: int = 100
    latest_turn_labour_index: int = 0
    current_task_score: int = 100
    current_task_labour_index: int = 0
    current_tok_mode: TokMode = TokMode.FULL_TOK
    event_counts: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.latest_turn_score = 100
        self.latest_turn_labour_index = 0
        self.current_task_score = 100
        self.current_task_labour_index = 0
        self.current_tok_mode = TokMode.FULL_TOK
        self.event_counts.clear()
