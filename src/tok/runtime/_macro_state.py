from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MacroState:
    load_global_macros: bool = True
    pending_heal: str = ""
    pending_heal_turn: int = 0

    def reset(self) -> None:
        self.pending_heal = ""
        self.pending_heal_turn = 0
