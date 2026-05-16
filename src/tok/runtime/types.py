"""Data models and types for the Tok runtime."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SurfaceMetadata(BaseModel, frozen=True):
    """Runtime-neutral identity and wire-shape metadata for an application surface."""

    runtime: str
    adapter: str
    input_shape: str = "anthropic_messages"
    output_shape: str = "anthropic_messages"
    supports_tool_pairs: bool = False
    uses_bridge_profile: bool = False
    requires_provider_canonicalization: bool = False
    uses_cut_search: bool = False
    uses_plan_finalization_guard: bool = False
    uses_first_turn_broad_audit_guard: bool = False
    model_config = {"extra": "forbid"}

    @classmethod
    def claude_bridge(cls) -> SurfaceMetadata:
        return cls(
            runtime="claude-code",
            adapter="claude-bridge",
            supports_tool_pairs=True,
            uses_bridge_profile=True,
            requires_provider_canonicalization=True,
            uses_cut_search=True,
            uses_plan_finalization_guard=True,
            uses_first_turn_broad_audit_guard=True,
        )

    @classmethod
    def from_adapter_kind(cls, adapter_kind: str) -> SurfaceMetadata:
        if adapter_kind == "claude-bridge":
            return cls.claude_bridge()
        if adapter_kind == "orchestrator":
            return cls(
                runtime="tok-orchestrator",
                adapter=adapter_kind,
                supports_tool_pairs=True,
                uses_bridge_profile=True,
                requires_provider_canonicalization=True,
            )
        return cls(runtime=adapter_kind or "unknown", adapter=adapter_kind or "unknown")

    def observability_fields(self) -> dict[str, str]:
        return {
            "surface_runtime": self.runtime,
            "surface_adapter": self.adapter,
            "surface_input_shape": self.input_shape,
            "surface_output_shape": self.output_shape,
        }


class EpisodeEntry(BaseModel, frozen=True):
    """
    A single completed reasoning episode.

    An episode spans the arc from a goal statement to a confirmed outcome
    (success, failure, or hand-off).  Entries are stored in the ledger and
    projected into working memory so the model doesn't re-open closed problems.
    """

    goal: str
    outcome: Literal["success", "failure", "partial", "open"]
    learnings: str  # one-line causal summary ("X failed because Y; fixed by Z")
    artifacts: list[str] = Field(default_factory=list)  # key files/commands touched


class EpisodeLedger(BaseModel):
    """
    Lightweight in-session episode log stored alongside bridge memory.

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
    args: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
    command: str | None = None
    query: str | None = None
    compressibility_class: Literal["raw", "file_read", "search", "command", "tool_result"] = "raw"
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
    surface: SurfaceMetadata | None = None
    tool_compatible: bool = False
    request_policy: Literal["legacy_tool_compatible", "natural_first", "forced_baseline"] = "legacy_tool_compatible"
    request_has_tools: bool = False
    allowed_tools: tuple[str, ...] | None = None
    grammar: str | None = None
    todo: str | None = None
    deltas: str | None = None
    model_config = {"extra": "forbid"}

    @property
    def surface_metadata(self) -> SurfaceMetadata:
        return self.surface or SurfaceMetadata.from_adapter_kind(self.adapter_kind)

    @property
    def surface_runtime(self) -> str:
        return self.surface_metadata.runtime

    @property
    def surface_adapter(self) -> str:
        return self.surface_metadata.adapter

    @property
    def uses_bridge_profile(self) -> bool:
        return self.surface_metadata.uses_bridge_profile

    @property
    def supports_tool_pairs(self) -> bool:
        return self.surface_metadata.supports_tool_pairs

    @property
    def requires_provider_canonicalization(self) -> bool:
        return self.surface_metadata.requires_provider_canonicalization

    @property
    def uses_cut_search(self) -> bool:
        return self.surface_metadata.uses_cut_search

    @property
    def uses_plan_finalization_guard(self) -> bool:
        return self.surface_metadata.uses_plan_finalization_guard

    @property
    def uses_first_turn_broad_audit_guard(self) -> bool:
        return self.surface_metadata.uses_first_turn_broad_audit_guard


class RecoveryAnchor(BaseModel, frozen=True):
    """Exact recovery pointer created by the core path before compact output is trusted."""

    kind: str
    key: str
    digest: str = ""
    source: str = ""
    model_config = {"extra": "forbid"}


class SafetyDecision(BaseModel, frozen=True):
    """Core-owned safety outcome adapters may observe but must not override."""

    allowed: bool = True
    reason: str = "allowed"
    fallback_required: bool = False
    exact_recovery_required: bool = False
    model_config = {"extra": "forbid"}


class SignalPacket(BaseModel, frozen=True):
    """Normalized packet handed from a surface adapter into Tok's core path."""

    request: RuntimeRequest
    recovery_anchors: list[RecoveryAnchor] = Field(default_factory=list)
    safety_decision: SafetyDecision = Field(default_factory=SafetyDecision)
    observability: dict[str, str | int | bool] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    @classmethod
    def from_request(cls, request: RuntimeRequest) -> SignalPacket:
        surface = request.surface_metadata
        observability: dict[str, str | int | bool] = dict(surface.observability_fields())
        observability["core_path"] = "runtime.prepare_request"
        observability["uses_bridge_profile"] = surface.uses_bridge_profile
        return cls(request=request, observability=observability)


class PreparedRuntimeRequest(BaseModel, frozen=True):
    """A prepared runtime request with metadata about the preparation process."""

    body: dict[str, Any]
    surface: SurfaceMetadata = Field(default_factory=lambda: SurfaceMetadata.from_adapter_kind("unknown"))
    compressed: bool
    input_saved_tokens: int
    behavior_signals: dict[str, int]
    type_breakdown: dict[str, int]
    mode: str
    request_policy: Literal["legacy_tool_compatible", "natural_first", "forced_baseline"] = "legacy_tool_compatible"
    effective_tool_compatible: bool = False
    request_policy_escalated: bool = False
    normalized_tool_events: list[NormalizedToolEvent] = Field(default_factory=list)
    baseline_prompt_tokens: int = 0
    prepared_prompt_tokens: int = 0
    saved_prompt_tokens: int = 0
    hot_hint_tokens_added: int = 0
    reacquisition_tokens_avoided_estimate: int = 0
    # Bloat attribution is a complex nested dict with mixed types
    # Using precise union type instead of Any
    bloat_attribution: dict[str, int | str | bool | dict[str, Any]] = Field(default_factory=dict)
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
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "RecoveryAnchor",
    "ReplayGateResult",
    "RuntimeRequest",
    "SafetyDecision",
    "SignalPacket",
    "SurfaceMetadata",
]
