"""
Memory management and retrieval for neural components.

This module provides memory structures and retrieval policies for
the neural components of the Tok system, including episodic memory,
rule storage, and lesson learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class TokMemory:
    tokens: frozenset[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tokens={len(self.tokens)})"


@dataclass(frozen=True, kw_only=True)
class EpisodeMemory(TokMemory):
    question: str
    answer: str
    ok: bool | None


@dataclass(frozen=True, kw_only=True)
class LessonMemory(TokMemory):
    lesson: str


@dataclass(frozen=True, kw_only=True)
class ConstraintMemory(TokMemory):
    """Stores negative knowledge or 'never-do' patterns."""

    constraint: str


@dataclass(frozen=True, kw_only=True)
class RepairMemory(TokMemory):
    # list of (code, error) pairs
    history: tuple[tuple[str, str], ...]
    final_ok: bool
