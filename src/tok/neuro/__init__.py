"""Backward-compatibility shim — import from tok.macros instead."""

from __future__ import annotations

from tok.macros.integration import distill_bridge_history
from tok.macros.ir import (
    Instruction,
    Macro,
    MacroProvenance,
    MacroRegistry,
    TokIR,
    execute_ir,
)
from tok.macros.memory import (
    ConstraintMemory,
    EpisodeMemory,
    LessonMemory,
    RepairMemory,
    TokMemory,
)
from tok.macros.miner import IRPatternMiner

__all__ = [
    "distill_bridge_history",
    "Instruction",
    "Macro",
    "MacroProvenance",
    "MacroRegistry",
    "TokIR",
    "execute_ir",
    "ConstraintMemory",
    "EpisodeMemory",
    "LessonMemory",
    "RepairMemory",
    "TokMemory",
    "IRPatternMiner",
]
