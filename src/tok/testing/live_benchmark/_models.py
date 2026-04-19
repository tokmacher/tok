from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    fixture_path: Path
    system_prompt: str
    followup_prompt: str
    success_terms: tuple[str, ...]
    min_success_terms: int = 2
    expected_file_terms: tuple[str, ...] = ()
    expected_verification_terms: tuple[str, ...] = ()
    require_file_field: bool | None = None
    require_verification_field: bool | None = None
    default_turns: int = 3
    prompt_sequence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.require_file_field is None:
            object.__setattr__(self, "require_file_field", bool(self.expected_file_terms))
        if self.require_verification_field is None:
            object.__setattr__(self, "require_verification_field", bool(self.expected_verification_terms))


@dataclass(frozen=True)
class ProviderUsageSnapshot:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float | None = None


@dataclass(frozen=True)
class ConversationTurnResult:
    mode: str
    provider_usage: ProviderUsageSnapshot
    compression_metrics: dict[str, Any]
    prompt_metrics: dict[str, Any]
    response_metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    outbound_payload: dict[str, Any]
    raw_response: str
    visible_response: str
    content_blocks: list[dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    mode: str
    model: str
    provider: str
    fixture_path: str
    provider_usage: ProviderUsageSnapshot
    compression_metrics: dict[str, Any]
    prompt_metrics: dict[str, Any]
    response_metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    task_success: bool
    matched_success_terms: list[str]
    request_messages: int
    turn_count: int
    turns: list[dict[str, Any]]
    visible_response: str
    raw_response: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provider_usage"] = asdict(self.provider_usage)
        return data


@dataclass(frozen=True)
class BenchmarkComparison:
    benchmark: str
    model: str
    candidate_mode: str
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    prompt_token_delta: int
    completion_token_delta: int
    total_token_delta: int
    total_token_delta_pct: float | None
    latency_delta_ms: float
    reacquisition_delta_tokens: int
    pressure_delta: int
    task_success_equal_or_better: bool
    provider_total_token_winner: str
    provider_cost_winner: str
    baseline_cost_usd: float | None
    candidate_cost_usd: float | None
    cost_delta_usd: float | None
    cost_delta_pct: float | None
    token_savings_without_cost_savings: bool
    cost_savings_without_token_savings: bool
    fairness_diagnostics: dict[str, Any]
    diagnosis: str
    tok_improved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "candidate_mode": self.candidate_mode,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "prompt_token_delta": self.prompt_token_delta,
            "completion_token_delta": self.completion_token_delta,
            "total_token_delta": self.total_token_delta,
            "total_token_delta_pct": self.total_token_delta_pct,
            "latency_delta_ms": self.latency_delta_ms,
            "reacquisition_delta_tokens": self.reacquisition_delta_tokens,
            "pressure_delta": self.pressure_delta,
            "task_success_equal_or_better": self.task_success_equal_or_better,
            "provider_total_token_winner": self.provider_total_token_winner,
            "provider_cost_winner": self.provider_cost_winner,
            "baseline_cost_usd": self.baseline_cost_usd,
            "candidate_cost_usd": self.candidate_cost_usd,
            "cost_delta_usd": self.cost_delta_usd,
            "cost_delta_pct": self.cost_delta_pct,
            "token_savings_without_cost_savings": self.token_savings_without_cost_savings,
            "cost_savings_without_token_savings": self.cost_savings_without_token_savings,
            "fairness_diagnostics": dict(self.fairness_diagnostics),
            "diagnosis": self.diagnosis,
            "tok_improved": self.tok_improved,
        }
