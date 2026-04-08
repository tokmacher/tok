"""Shared models and constants for the internal stress harness package."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from tok.utils.config import API_BASE

if TYPE_CHECKING:
    from pathlib import Path

READ_ONLY_TOOL_NAMES = {
    "view_file",
    "read",
    "grep_search",
    "search",
    "grep",
    "rg",
    "list_dir",
    "ls",
}

LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS = 800
PRE_PRESSURE_MIN_EVIDENCE_CHARS = 500
LATE_STAGED_RETRY_PHASES = {
    "tool-contract",
    "fresh-grounding",
    "payload-pressure",
    "near-neighbor disambiguation",
}

_PATH_PATTERN = re.compile(r"(src/[\w./-]+\.\w+|tests/[\w./-]+\.\w+|docs/[\w./-]+\.\w+)")
EXCLUDED_GROUNDED_PATH_FRAGMENTS = (
    "src/tok/stress_harness.py",
    "tests/",
    "tmp/stress_language/",
)
DEFAULT_REQUIRED_CLASSES = (
    "tool_contract_failure",
    "retention_loss",
    "reacquisition_loop",
    "protocol_drift",
    "compaction_loss|baseline_fallback",
)


@dataclass(frozen=True)
class StressTask:
    id: str
    phase_name: str
    prompt: str
    expected_file: str = ""
    expected_verification: str = ""
    require_fresh_evidence: bool = False
    require_tool_count: int = 0
    required_tool_names: tuple[str, ...] = ()
    forbid_reacquisition: bool = False
    dynamic_anchor: str = ""
    min_validated_anchors: int = 0
    min_reuse_checks: int = 0
    min_checkpoint_checks: int = 0
    force_payload: bool = False
    requires_memory_surfaces: bool = False


@dataclass(frozen=True)
class StressHarnessConfig:
    model: str = "qwen/qwen3-coder-next"
    provider: str = "openrouter"
    api_key: str | None = None
    api_base: str = API_BASE
    temperature: float = 0.0
    max_tokens: int = 450
    target_breakpoints: int = 5
    max_tasks: int = 24
    max_tool_rounds: int = 8
    max_retries_per_task: int = 2
    min_payload_pressure_bytes: int = 12000
    fallback_threshold: int = int(os.getenv("TOK_FALLBACK_THRESHOLD", "3"))
    output_dir: Path | None = None
    provider_options: dict[str, Any] | None = None
    progress: bool = False
    required_classes: tuple[str, ...] = DEFAULT_REQUIRED_CLASSES
    task_catalog: tuple[StressTask, ...] | None = None


@dataclass(frozen=True)
class ValidatedAnchor:
    task_id: str
    phase_name: str
    file: str
    verification: str
    turn_index: int
    evidence_chars: int

    def to_fields(self) -> dict[str, str]:
        return {"file": self.file, "verification": self.verification}


@dataclass(frozen=True)
class StressBreakpoint:
    breakpoint_class: str
    task_id: str
    turn_index: int
    prompt: str
    visible_response: str
    active_tools: list[str]
    input_behavior_signals: dict[str, int]
    output_behavior_signals: dict[str, int]
    state_payload_chars: int
    resend_mode: str
    transcript_slice: list[dict[str, Any]]
    inferred_cause: str
    refactor_target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StressTurnRecord:
    task_id: str
    phase_name: str
    turn_index: int
    phase: str
    prompt: str
    raw_response: str
    visible_response: str
    tool_uses: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    evidence_chars: int
    retry_index: int
    validated: bool
    input_behavior_signals: dict[str, int]
    output_behavior_signals: dict[str, int]
    input_saved_tokens: int
    output_saved_tokens: int
    tool_contract_failure: bool
    state_payload_chars: int
    resend_mode: str
    resend_decision_reason: str = ""
    memory_loaded_chars: int = 0
    tool_result_volume_chars: int = 0
    tool_dense_session: bool = False
    answer_fact_projection_present: bool = False
    payload_pressure_ready: bool = False
    compaction_eligible_ready: bool = False
    validated_target_reacquired: bool = False
    validated_target_exact_reacquired: bool = False
    validated_target_reconfirmation_attempt: bool = False
    answer_anchor_reacquisition_attempt: bool = False
    answer_ready_reacquisition_attempt: bool = False
    repair_phase_reacquisition_attempt: bool = False
    benign_reverification_attempt: bool = False
    request_messages: int = 0
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    baseline_only: bool = False
    task_completed_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StressRunResult:
    model: str
    provider: str
    started_at: str
    completed_at: str
    target_breakpoints: int
    required_classes: tuple[str, ...]
    max_tasks: int
    max_tool_rounds: int
    tasks_completed: int
    baseline_only: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    validated_anchor_count: int
    tool_backed_turns: int
    resend_modes_seen: list[str]
    payload_pressure_reached: bool
    compaction_eligible: bool
    reuse_checks_run: int
    checkpoint_checks_run: int
    reuse_probe_attempts: int
    reuse_probe_successes: int
    retention_probe_attempts: int
    retention_probe_successes: int
    late_retention_probe_attempts: int
    late_retention_probe_successes: int
    tool_contract_probe_attempts: int
    tool_contract_failure_events_seen: int
    mixed_answer_tool_events_seen: int
    unsupported_tool_events_seen: int
    bad_tool_args_events_seen: int
    toolless_fresh_answer_events_seen: int
    reacquisition_events_seen: int
    validated_target_reacquisition_events_seen: int
    validated_target_exact_reacquisition_events_seen: int
    validated_target_reconfirmation_events_seen: int
    answer_anchor_reacquisition_events_seen: int
    answer_ready_reacquisition_events_seen: int
    repair_phase_reacquisition_events_seen: int
    benign_reverification_events_seen: int
    answer_ready_repair_requested_count: int
    answer_ready_repair_active_count: int
    answer_ready_repair_resolved_count: int
    answer_ready_repair_failed_count: int
    late_freshness_signal_promoted_count: int
    late_freshness_signal_consumed_by_tok_count: int
    late_mixed_signal_promoted_count: int
    late_mixed_signal_consumed_by_tok_count: int
    late_answer_assembly_repair_answer_only_requested_count: int
    late_answer_assembly_repair_answer_only_resolved_count: int
    late_answer_assembly_repair_answer_only_failed_count: int
    late_answer_followthrough_requested_count: int
    late_answer_followthrough_active_count: int
    late_answer_followthrough_resolved_count: int
    late_answer_followthrough_failed_count: int
    late_answer_followthrough_after_tool_only_repair_count: int
    late_answer_followthrough_blocked_insufficient_evidence_count: int
    late_tool_contract_reconfirmation_grace_count: int
    late_tool_contract_mixed_grace_count: int
    late_tool_contract_toolless_grace_count: int
    late_tool_contract_reconfirmation_retry_failure_count: int
    late_tool_contract_mixed_retry_failure_count: int
    late_tool_contract_toolless_retry_failure_count: int
    fallback_pressure_incremented_count: int
    fallback_pressure_suppressed_count: int
    fallback_pressure_cause_exact_reacquisition_count: int
    fallback_pressure_cause_mixed_turn_count: int
    fallback_pressure_cause_toolless_fresh_count: int
    fallback_pressure_cause_bad_args_count: int
    fallback_pressure_cause_unsupported_tool_count: int
    retry_prompt_shape_exact_target_reread_count: int
    retry_prompt_shape_mixed_turn_count: int
    retry_prompt_shape_toolless_fresh_count: int
    retry_prompt_shape_unsupported_tool_count: int
    retry_prompt_shape_bad_args_count: int
    retry_prompt_shape_generic_retry_count: int
    retry_prompt_no_exact_reread_count: int
    retry_prompt_requires_supporting_tool_count: int
    retry_prompt_supporting_tool_satisfied_count: int
    retry_prompt_supporting_tool_missed_count: int
    retry_prompt_supporting_tool_missed_mixed_count: int
    retry_prompt_supporting_tool_missed_toolless_count: int
    exact_target_reread_after_no_exact_retry_count: int
    early_retry_contract_stage_tool_only_count: int
    early_retry_contract_stage_answer_only_count: int
    early_retry_bad_args_tool_only_count: int
    early_retry_tool_only_satisfied_count: int
    early_retry_tool_only_failed_mixed_count: int
    early_retry_tool_only_failed_toolless_count: int
    early_retry_answer_only_satisfied_count: int
    early_retry_answer_only_failed_tool_count: int
    late_retry_contract_stage_tool_only_count: int
    late_retry_contract_stage_answer_only_count: int
    late_retry_tool_only_satisfied_count: int
    late_retry_tool_only_failed_mixed_count: int
    late_retry_tool_only_failed_toolless_count: int
    late_retry_answer_only_satisfied_count: int
    late_retry_answer_only_failed_tool_count: int
    late_retry_no_exact_target_count: int
    exact_target_reread_after_late_retry_no_exact_target_count: int
    failed_tasks_before_any_retry_contract_count: int
    failed_tasks_after_generic_retry_only_count: int
    failed_tasks_after_early_staged_retry_count: int
    failed_tasks_after_late_staged_retry_count: int
    failed_tasks_after_validated_target_retry_count: int
    first_failed_phase: str
    first_failed_task: str
    first_irreversible_miss_kind: str
    dominant_failure_locus: str
    first_payload_pressure_turn: int | None
    first_payload_pressure_task: str
    first_compaction_eligible_turn: int | None
    first_compaction_eligible_task: str
    first_baseline_fallback_turn: int | None
    first_baseline_fallback_task: str
    baseline_fallback_turns_after_payload_pressure: int
    baseline_fallback_turns_after_compaction_eligible: int
    fallback_after_payload_pressure: bool
    fallback_after_compaction_eligible: bool
    retention_substitution_events_seen: int
    compaction_eligible_turns: int
    anchors_before_baseline: int
    seed_searches: int
    seed_direct_reads: int
    seed_answer_attempts: int
    seed_evidence_sufficient: bool
    first_anchor_failure_mode: str
    run_diagnosis: str
    weak_run_reasons: list[str]
    breakpoints: list[StressBreakpoint]
    turns: list[StressTurnRecord]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["breakpoints"] = [item.to_dict() for item in self.breakpoints]
        data["turns"] = [item.to_dict() for item in self.turns]
        return data


@dataclass(frozen=True)
class StressObservation:
    task_id: str
    turn_index: int
    prompt: str
    phase: str
    visible_response: str
    active_tools: list[str]
    input_behavior_signals: dict[str, int]
    output_behavior_signals: dict[str, int]
    state_payload_chars: int
    resend_mode: str
    transcript_slice: list[dict[str, Any]]
    expected_fields: dict[str, str] = field(default_factory=dict)
    observed_fields: dict[str, str] = field(default_factory=dict)
    baseline_only: bool = False
    tool_contract_failure: bool = False
    repeated_oracle_miss: bool = False
    validated_target_reacquired: bool = False
    validated_target_exact_reacquired: bool = False
    validated_target_reconfirmation_attempt: bool = False
    target_already_validated: bool = False
    payload_pressure_ready: bool = False
    seed_evidence_sufficient: bool = False
    repeated_seed_search_without_read: bool = False
    repeated_seed_tool_after_evidence: bool = False
    retention_latest_substitution: bool = False
