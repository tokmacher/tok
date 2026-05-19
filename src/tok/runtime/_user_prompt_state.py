from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserPromptState:
    last_text: str = ""
    last_labels: tuple[str, ...] = ()
    request_has_tools: bool = False
    hint_last_turn: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_text = ""
        self.last_labels = ()
        self.request_has_tools = False
        self.hint_last_turn.clear()
