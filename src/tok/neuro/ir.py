"""Backward-compatibility shim — import from tok.macros.ir instead."""

from __future__ import annotations

from tok.macros.ir import (
    Instruction,
    Macro,
    MacroProvenance,
    MacroRegistry,
    TokIR,
    execute_ir,
)

__all__ = [
    "Instruction",
    "Macro",
    "MacroProvenance",
    "MacroRegistry",
    "TokIR",
    "execute_ir",
]
