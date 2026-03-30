"""Data models and types for the Tok runtime."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EpisodeEntry(BaseModel, frozen=True):
    """A single completed reasoning episode.

    An episode spans the arc from a goal statement to a confirmed outcome
    (success, failure, or hand-off).  Entries are stored in the ledger and
    projected into working memory so the model doesn't re-open closed problems.
    """

    goal: str
    outcome: Literal["success", "failure", "partial", "open"]
    learnings: (
        str  # one-line causal summary ("X failed because Y; fixed by Z")
    )
    artifacts: list[str] = Field(
        default_factory=list
    )  # key files/commands touched


class EpisodeLedger(BaseModel):
    """Lightweight in-session episode log stored alongside bridge memory.

    Keeps the last `max_entries` episodes.  Older entries are dropped to bound
    memory size while preserving the most recent learning chain.
    """

    entries: list[EpisodeEntry] = Field(default_factory=list)
    max_entries: int = 8
    model_config = {"extra": "forbid"}

    def record(self, entry: EpisodeEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def wire_state(self) -> str:
        """Compact projection of episodes for injection into the Tok state line."""
        if not self.entries:
            return ""
        parts = []
        for e in self.entries[-3:]:  # project last 3 episodes
            outcome_tag = e.outcome[:1].upper()  # S/F/P/O
            parts.append(f"{e.goal[:24]}:{outcome_tag}:{e.learnings[:32]}")
        return "episodes:" + "|".join(parts)

    def to_tok(self) -> str:
        """Serialize ledger to a compact text format for persistence."""
        lines = ["@episode_ledger"]
        for e in self.entries:
            lines.append(
                json.dumps(
                    {
                        "goal": e.goal,
                        "outcome": e.outcome,
                        "learnings": e.learnings,
                        "artifacts": e.artifacts,
                    }
                )
            )
        return "\n".join(lines)

    @classmethod
    def from_tok(cls, text: str) -> EpisodeLedger:
        """Deserialize from the persisted text format."""
        ledger = cls()
        in_ledger = False
        for line in text.splitlines():
            s = line.strip()
            if s == "@episode_ledger":
                in_ledger = True
                continue
            if not in_ledger or not s:
                continue
            try:
                data = json.loads(s)
                ledger.entries.append(
                    EpisodeEntry(
                        goal=data.get("goal", ""),
                        outcome=data.get("outcome", "open"),
                        learnings=data.get("learnings", ""),
                        artifacts=data.get("artifacts", []),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return ledger


class NormalizedToolEvent(BaseModel, frozen=True):
    """A normalized representation of a tool use event."""

    id: str
    name: str
    args: dict[
        str,
        str
        | int
        | float
        | bool
        | list[str | int | float | bool]
        | dict[str, str | int | float | bool]
        | None,
    ] = Field(default_factory=dict)
    path: str | None = None
    command: str | None = None
    query: str | None = None
    compressibility_class: Literal[
        "raw", "file_read", "search", "command", "tool_result"
    ] = "raw"
    fidelity_requirement: str = "default"
    model_config = {"extra": "forbid"}

    @field_validator("compressibility_class")
    @classmethod
    def validate_compressibility_class(cls, v: str) -> str:
        """Validate compressibility_class is a known value."""
        if v not in ("raw", "file_read", "search", "command", "tool_result"):
            return "raw"  # Default to raw for unknown values
        return v


class RuntimeRequest(BaseModel, frozen=True):
    """A runtime request to be prepared for the model."""

    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    adapter_kind: str = "unknown"
    tool_compatible: bool = False
    grammar: str | None = None
    todo: str | None = None
    deltas: str | None = None
    model_config = {"extra": "forbid"}


class PreparedRuntimeRequest(BaseModel, frozen=True):
    """A prepared runtime request with metadata about the preparation process."""

    body: dict[str, Any]
    compressed: bool
    input_saved_tokens: int
    behavior_signals: dict[str, int]
    type_breakdown: dict[str, int]
    mode: str
    normalized_tool_events: list[NormalizedToolEvent] = Field(
        default_factory=list
    )
    baseline_prompt_tokens: int = 0
    prepared_prompt_tokens: int = 0
    saved_prompt_tokens: int = 0
    hot_hint_tokens_added: int = 0
    reacquisition_tokens_avoided_estimate: int = 0
    # Bloat attribution is a complex nested dict with mixed types
    # Using precise union type instead of Any
    bloat_attribution: dict[str, int | str | bool | dict[str, Any]] = Field(
        default_factory=dict
    )
    model_config = {"extra": "forbid"}


class ProcessedRuntimeResponse(BaseModel, frozen=True):
    """A processed runtime response with metadata about the response processing."""

    content_blocks: list[dict[str, Any]]
    output_saved_tokens: int
    behavior_signals: dict[str, int]
    mode: str
    family_mode: str
    updated_memory: str
    model_config = {"extra": "forbid"}


class ReplayGateResult(BaseModel, frozen=True):
    """Result of evaluating whether a replay should be granted."""

    passed: bool
    invisible_pressure: int
    failed_checks: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}


__all__ = [
    "EpisodeEntry",
    "EpisodeLedger",
    "NormalizedToolEvent",
    "RuntimeRequest",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "ReplayGateResult",
]
