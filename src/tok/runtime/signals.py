"""Internal registry for behavior signals that matter to diagnostics and release gates."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

_VALID_CATEGORIES = frozenset(
    {
        "fallback",
        "recovery",
        "provider_safety",
        "evidence_safety",
        "compression",
        "session",
        "trace",
    }
)
_VALID_SEVERITIES = frozenset({"info", "warning", "critical"})

logger = logging.getLogger("tok.signals")


@dataclass(frozen=True)
class SignalDefinition:
    """Metadata for behavior signals with release or diagnostic meaning."""

    name: str
    category: str
    severity: str
    label: str
    affects_health: bool = False
    release_critical: bool = False

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(
                f"Signal {self.name!r}: invalid category {self.category!r}. Must be one of {sorted(_VALID_CATEGORIES)}"
            )
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Signal {self.name!r}: invalid severity {self.severity!r}. Must be one of {sorted(_VALID_SEVERITIES)}"
            )


def _signal(
    name: str,
    *,
    category: str,
    severity: str,
    label: str,
    affects_health: bool = False,
    release_critical: bool = True,
) -> SignalDefinition:
    return SignalDefinition(
        name=name,
        category=category,
        severity=severity,
        label=label,
        affects_health=affects_health,
        release_critical=release_critical,
    )


_SIGNALS = (
    _signal(
        "tok_fallback_activated",
        category="fallback",
        severity="warning",
        label="Tok fallback activated",
        affects_health=True,
    ),
    _signal(
        "baseline_only_session",
        category="fallback",
        severity="warning",
        label="Baseline-only session",
        affects_health=True,
    ),
    _signal(
        "non_tok_response",
        category="fallback",
        severity="warning",
        label="Non-Tok response",
        affects_health=True,
    ),
    _signal(
        "fail_open_compat_response",
        category="fallback",
        severity="warning",
        label="Compatibility fallback",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_response",
        category="fallback",
        severity="warning",
        label="Malformed Tok response",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_hybrid_tool",
        category="fallback",
        severity="warning",
        label="Malformed hybrid tool response",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_non_inverted_msg",
        category="fallback",
        severity="warning",
        label="Malformed non-inverted message",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_markdown_fallback",
        category="fallback",
        severity="warning",
        label="Markdown fallback response",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_bad_header",
        category="fallback",
        severity="warning",
        label="Malformed Tok header",
        affects_health=True,
    ),
    _signal(
        "request_policy_escalations",
        category="recovery",
        severity="info",
        label="Request policy escalations",
    ),
    _signal(
        "request_policy_deescalations",
        category="recovery",
        severity="info",
        label="Request policy deescalations",
    ),
    _signal(
        "request_policy_reason_stream_recovery",
        category="recovery",
        severity="info",
        label="Stream recovery policy hold",
    ),
    _signal(
        "request_policy_reason_tool_recovery",
        category="recovery",
        severity="info",
        label="Tool recovery policy hold",
    ),
    _signal(
        "request_policy_reason_structured_tool_loop",
        category="recovery",
        severity="info",
        label="Structured tool-loop policy hold",
    ),
    _signal(
        "request_policy_held_by_recovery",
        category="recovery",
        severity="info",
        label="Policy held by recovery",
    ),
    _signal(
        "request_policy_recovery_sticky_continuations",
        category="recovery",
        severity="info",
        label="Sticky recovery continuations",
    ),
    _signal(
        "stream_recovery_started",
        category="recovery",
        severity="info",
        label="Stream recovery started",
    ),
    _signal(
        "stream_recovery_retry",
        category="recovery",
        severity="info",
        label="Stream recovery retry",
    ),
    _signal(
        "stream_recovery_empty_success",
        category="recovery",
        severity="info",
        label="Empty stream recovered",
    ),
    _signal(
        "stream_recovery_read_error",
        category="recovery",
        severity="info",
        label="Stream read error recovered",
    ),
    _signal(
        "stream_recovery_success_text",
        category="recovery",
        severity="info",
        label="Stream recovered as text",
    ),
    _signal(
        "stream_recovery_success_tool_use",
        category="recovery",
        severity="info",
        label="Stream recovered as tool use",
    ),
    _signal(
        "stream_recovery_fallback",
        category="recovery",
        severity="warning",
        label="Stream recovery fallback",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_provider_sensitive_degraded_to_provider_safe",
        category="provider_safety",
        severity="warning",
        label="Provider-sensitive request degraded safely",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_provider_sensitive_blocked_local",
        category="provider_safety",
        severity="warning",
        label="Provider-sensitive request blocked locally",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_provider_pairing_risk_detected",
        category="provider_safety",
        severity="warning",
        label="Provider pairing risk detected",
        affects_health=True,
    ),
    _signal(
        "fail_open_retry_upstream_pairing_disagreement",
        category="provider_safety",
        severity="warning",
        label="Upstream pairing disagreement",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_assistant_tool_use_text_interleaving_blocked",
        category="provider_safety",
        severity="warning",
        label="Assistant tool/text interleaving blocked",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_invalid_tool_history_blocked",
        category="provider_safety",
        severity="warning",
        label="Invalid tool history blocked",
        affects_health=True,
    ),
    _signal(
        "evidence_exact_observed",
        category="evidence_safety",
        severity="info",
        label="Exact evidence observed",
    ),
    _signal(
        "evidence_first_exact_observed",
        category="evidence_safety",
        severity="info",
        label="First exact evidence observed",
    ),
    _signal(
        "evidence_non_exact_reference_emitted",
        category="evidence_safety",
        severity="info",
        label="Non-exact evidence reference emitted",
    ),
    _signal(
        "evidence_non_exact_summary_emitted",
        category="evidence_safety",
        severity="info",
        label="Non-exact evidence summary emitted",
    ),
    _signal(
        "evidence_non_exact_skeleton_emitted",
        category="evidence_safety",
        severity="info",
        label="Non-exact evidence skeleton emitted",
    ),
    _signal(
        "evidence_exact_reacquisition_required",
        category="evidence_safety",
        severity="info",
        label="Exact evidence reacquisition required",
    ),
    _signal(
        "evidence_exact_reacquisition_satisfied",
        category="evidence_safety",
        severity="info",
        label="Exact evidence reacquisition satisfied",
    ),
    _signal(
        "evidence_compression_blocked_for_safety",
        category="evidence_safety",
        severity="info",
        label="Evidence compression blocked for safety",
    ),
    _signal(
        "evidence_resolver_cache_stored",
        category="evidence_safety",
        severity="info",
        label="Exact evidence stored in resolver cache",
    ),
    _signal(
        "evidence_tool_result_compression_skipped",
        category="evidence_safety",
        severity="info",
        label="Tool-result compression skipped for evidence safety",
    ),
    _signal(
        "evidence_history_compression_skipped",
        category="evidence_safety",
        severity="info",
        label="History compression skipped for evidence safety",
    ),
    # ---------------------------------------------------------------------------
    # Fallback / processing errors (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "processing_error",
        category="fallback",
        severity="warning",
        label="Bridge processing error",
        affects_health=True,
    ),
    _signal(
        "critical_system_error",
        category="fallback",
        severity="critical",
        label="Critical system error",
        affects_health=True,
    ),
    _signal(
        "data_structure_error",
        category="fallback",
        severity="warning",
        label="Data structure error",
        affects_health=True,
    ),
    _signal(
        "json_decode_error",
        category="fallback",
        severity="warning",
        label="JSON decode error",
        affects_health=True,
    ),
    _signal(
        "unexpected_error",
        category="fallback",
        severity="warning",
        label="Unexpected bridge error",
        affects_health=True,
    ),
    _signal(
        "streaming_error_retry",
        category="fallback",
        severity="warning",
        label="Streaming error with retry",
        affects_health=True,
    ),
    _signal(
        "stream_buffer_read_error",
        category="fallback",
        severity="warning",
        label="Stream buffer read error",
        affects_health=True,
    ),
    _signal(
        "tok_fail_open_retry",
        category="fallback",
        severity="warning",
        label="Tok fail-open retry",
        affects_health=True,
    ),
    _signal(
        "tok_fallback_zero_compression_revert",
        category="fallback",
        severity="warning",
        label="Zero-compression revert to baseline",
        affects_health=True,
    ),
    # ---------------------------------------------------------------------------
    # Recovery signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "stream_recovery_loop_broken",
        category="recovery",
        severity="warning",
        label="Stream recovery loop broken",
        affects_health=True,
    ),
    _signal(
        "stream_recovery_retry_exception",
        category="recovery",
        severity="warning",
        label="Stream recovery retry exception",
        affects_health=True,
    ),
    _signal(
        "stream_recovery_history_floor_applied",
        category="recovery",
        severity="info",
        label="Stream recovery history floor applied",
        release_critical=False,
    ),
    _signal(
        "stream_recovery_history_floor_kept_context",
        category="recovery",
        severity="info",
        label="Stream recovery kept context on floor",
        release_critical=False,
    ),
    _signal(
        "stream_recovery_history_floor_noop",
        category="recovery",
        severity="info",
        label="Stream recovery history floor noop",
        release_critical=False,
    ),
    _signal(
        "stream_recovery_usage",
        category="recovery",
        severity="info",
        label="Stream recovery usage recorded",
        release_critical=False,
    ),
    _signal(
        "stream_empty_after_success",
        category="recovery",
        severity="warning",
        label="Stream empty after apparent success",
        affects_health=True,
    ),
    # ---------------------------------------------------------------------------
    # Provider safety signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "tok_bridge_pairing_degraded_to_provider_safe",
        category="provider_safety",
        severity="warning",
        label="Bridge pairing degraded to provider-safe",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_thinking_mutation_degraded_to_provider_safe",
        category="provider_safety",
        severity="warning",
        label="Thinking mutation degraded to provider-safe",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_thinking_mutation_unrestored",
        category="provider_safety",
        severity="warning",
        label="Thinking mutation not restored",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_preflight_failed_local",
        category="provider_safety",
        severity="warning",
        label="Bridge preflight failed locally",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_preflight_rejected",
        category="provider_safety",
        severity="warning",
        label="Bridge preflight rejected",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_prepared_pairing_rejected_local",
        category="provider_safety",
        severity="warning",
        label="Prepared pairing rejected locally",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_tool_history_pairing_repaired",
        category="provider_safety",
        severity="info",
        label="Tool history pairing repaired",
    ),
    _signal(
        "tok_bridge_tool_history_repaired",
        category="provider_safety",
        severity="info",
        label="Tool history repaired",
    ),
    _signal(
        "tok_compression_worked_before_pairing_degraded",
        category="provider_safety",
        severity="warning",
        label="Compression worked but pairing later degraded",
        affects_health=True,
    ),
    _signal(
        "tok_history_pairing_safety_degraded",
        category="provider_safety",
        severity="warning",
        label="History pairing safety degraded",
        affects_health=True,
    ),
    _signal(
        "tok_release_blocking_thinking_mutation",
        category="provider_safety",
        severity="critical",
        label="Release-blocking thinking mutation",
        affects_health=True,
    ),
    _signal(
        "preflight_block_rewritten_payload",
        category="provider_safety",
        severity="info",
        label="Preflight block rewrote payload",
    ),
    # ---------------------------------------------------------------------------
    # Session / request policy signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "request_policy_forced_baseline",
        category="session",
        severity="warning",
        label="Request policy forced to baseline",
        affects_health=True,
    ),
    _signal(
        "request_policy_natural_first",
        category="session",
        severity="info",
        label="Request policy: natural-first",
        release_critical=False,
    ),
    _signal(
        "request_policy_tool_compatible",
        category="session",
        severity="info",
        label="Request policy: tool-compatible",
        release_critical=False,
    ),
    _signal(
        "request_policy_effective_tool_compatible",
        category="session",
        severity="info",
        label="Effective request policy: tool-compatible",
        release_critical=False,
    ),
    _signal(
        "request_policy_effective_natural_first",
        category="session",
        severity="info",
        label="Effective request policy: natural-first",
        release_critical=False,
    ),
    _signal(
        "request_policy_transition_to_tool_compatible",
        category="session",
        severity="info",
        label="Policy transition to tool-compatible",
        release_critical=False,
    ),
    _signal(
        "request_policy_transition_to_natural_first",
        category="session",
        severity="info",
        label="Policy transition to natural-first",
        release_critical=False,
    ),
    _signal(
        "request_policy_transition_unchanged",
        category="session",
        severity="info",
        label="Policy transition unchanged",
        release_critical=False,
    ),
    _signal(
        "request_policy_requested_tool_compatible",
        category="session",
        severity="info",
        label="Requested policy: tool-compatible",
        release_critical=False,
    ),
    _signal(
        "request_policy_requested_non_tool_compatible",
        category="session",
        severity="info",
        label="Requested policy: non-tool-compatible",
        release_critical=False,
    ),
    _signal(
        "request_policy_requested_natural_first_effective_tool_compatible",
        category="session",
        severity="info",
        label="Natural-first requested but tool-compatible effective",
        release_critical=False,
    ),
    _signal(
        "request_policy_recovery_cooldown_suppressed",
        category="session",
        severity="info",
        label="Policy recovery cooldown suppressed",
        release_critical=False,
    ),
    _signal(
        "request_policy_interleaving_downgrades",
        category="session",
        severity="info",
        label="Policy interleaving downgrades",
        release_critical=False,
    ),
    _signal(
        "short_session_baseline_mode",
        category="session",
        severity="info",
        label="Short session baseline mode",
        release_critical=False,
    ),
    _signal(
        "thinking_forced_non_stream",
        category="session",
        severity="info",
        label="Thinking forced non-streaming",
        release_critical=False,
    ),
    _signal(
        "smoothness_streaming_disabled",
        category="session",
        severity="info",
        label="Smoothness streaming disabled",
        release_critical=False,
    ),
    _signal(
        "jit_offer_available",
        category="session",
        severity="info",
        label="JIT offer available",
        release_critical=False,
    ),
    _signal(
        "jit_offer_context_filtered",
        category="session",
        severity="info",
        label="JIT offer context filtered",
        release_critical=False,
    ),
    _signal(
        "repeat_command_suppression_hint_injected",
        category="session",
        severity="info",
        label="Repeat command suppression hint injected",
        release_critical=False,
    ),
    _signal(
        "loop_terminated",
        category="session",
        severity="info",
        label="Loop terminated",
        release_critical=False,
    ),
    # ---------------------------------------------------------------------------
    # Compression signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "lossless_task_mode_history_skipped",
        category="compression",
        severity="info",
        label="Lossless task mode: history compression skipped",
        release_critical=False,
    ),
    _signal(
        "broad_audit_history_skipped",
        category="compression",
        severity="info",
        label="Broad-audit: history compression skipped",
        release_critical=False,
    ),
    _signal(
        "broad_audit_tok_additions_suppressed",
        category="compression",
        severity="info",
        label="Broad-audit: Tok additions suppressed",
        release_critical=False,
    ),
    _signal(
        "broad_audit_tool_result_compression_skipped",
        category="compression",
        severity="info",
        label="Broad-audit: tool-result compression skipped",
        release_critical=False,
    ),
    _signal(
        "tok_history_compression_skipped",
        category="compression",
        severity="info",
        label="History compression skipped",
        release_critical=False,
    ),
    _signal(
        "tok_history_cut_blocked_tool_result",
        category="compression",
        severity="info",
        label="History cut blocked by tool result",
        release_critical=False,
    ),
    _signal(
        "tok_history_cut_point_missing",
        category="compression",
        severity="info",
        label="History cut point missing",
    ),
    _signal(
        "tok_history_cut_point_missing_with_tools",
        category="compression",
        severity="info",
        label="History cut point missing with tools present",
    ),
    _signal(
        "compress_tool_results_bypassed",
        category="compression",
        severity="info",
        label="Tool-result compression bypassed",
        release_critical=False,
    ),
    _signal(
        "tool_compatible_compression",
        category="compression",
        severity="info",
        label="Tool-compatible compression applied",
        release_critical=False,
    ),
    _signal(
        "tok_context_compression_detected",
        category="compression",
        severity="info",
        label="Context compression detected",
        release_critical=False,
    ),
    _signal(
        "tok_prompt_bloat_detected",
        category="compression",
        severity="info",
        label="Prompt bloat detected",
        release_critical=False,
    ),
    _signal(
        "tok_prompt_optimized",
        category="compression",
        severity="info",
        label="Prompt optimized",
        release_critical=False,
    ),
    _signal(
        "tok_prompt_optimization_blocked",
        category="compression",
        severity="info",
        label="Prompt optimization blocked",
        release_critical=False,
    ),
    _signal(
        "tok_prompt_optimization_skipped_bridge",
        category="compression",
        severity="info",
        label="Prompt optimization skipped (bridge)",
        release_critical=False,
    ),
    _signal(
        "bridge_history_cut_search_used",
        category="compression",
        severity="info",
        label="History cut extended to search boundary",
        release_critical=False,
    ),
    _signal(
        "bridge_history_cut_search_extended",
        category="compression",
        severity="info",
        label="History cut search extended",
        release_critical=False,
    ),
    _signal(
        "plan_finalization_turn",
        category="compression",
        severity="info",
        label="Plan finalization turn",
    ),
    _signal(
        "plan_finalization_history_skipped",
        category="compression",
        severity="info",
        label="Plan finalization: history compression skipped",
        release_critical=False,
    ),
    _signal(
        "plan_finalization_tool_result_compression_skipped",
        category="compression",
        severity="info",
        label="Plan finalization: tool-result compression skipped",
        release_critical=False,
    ),
    _signal(
        "plan_finalization_tok_overhead_blocked",
        category="compression",
        severity="info",
        label="Plan finalization: Tok overhead blocked",
        release_critical=False,
    ),
    _signal(
        "plan_finalization_tool_escalation_suppressed",
        category="compression",
        severity="info",
        label="Plan finalization: tool escalation suppressed",
        release_critical=False,
    ),
    _signal(
        "plan_finalization_passthrough",
        category="compression",
        severity="info",
        label="Plan finalization: passthrough",
        release_critical=False,
    ),
    _signal(
        "retention_latest_substitution",
        category="compression",
        severity="info",
        label="Retention latest substitution applied",
        release_critical=False,
    ),
    _signal(
        "tok_sift_cache_marked_blocks",
        category="compression",
        severity="info",
        label="Sift cache-marked blocks",
        release_critical=False,
    ),
    _signal(
        "client_cache_marker_dropped",
        category="compression",
        severity="info",
        label="Client cache marker dropped",
        release_critical=False,
    ),
    # ---------------------------------------------------------------------------
    # Answer-phase and evidence signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "answer_ready_turn",
        category="evidence_safety",
        severity="info",
        label="Answer-ready turn",
    ),
    _signal(
        "answer_anchor_present",
        category="evidence_safety",
        severity="info",
        label="Answer anchor present",
    ),
    _signal(
        "answer_anchor_reacquisition_attempt",
        category="evidence_safety",
        severity="info",
        label="Answer anchor reacquisition attempt",
    ),
    _signal(
        "answer_ready_reacquisition_attempt",
        category="evidence_safety",
        severity="info",
        label="Answer-ready reacquisition attempt",
    ),
    _signal(
        "exploration_reacquisition_expected",
        category="evidence_safety",
        severity="info",
        label="Exploration reacquisition expected",
        release_critical=False,
    ),
    _signal(
        "repair_phase_reacquisition_attempt",
        category="evidence_safety",
        severity="info",
        label="Repair phase reacquisition attempt",
    ),
    _signal(
        "benign_reverification_attempt",
        category="evidence_safety",
        severity="info",
        label="Benign reverification attempt",
        release_critical=False,
    ),
    _signal(
        "answer_ready_repair_active",
        category="evidence_safety",
        severity="info",
        label="Answer-ready repair active",
    ),
    _signal(
        "late_answer_assembly_repair_active",
        category="evidence_safety",
        severity="info",
        label="Late answer assembly repair active",
    ),
    _signal(
        "late_answer_followthrough_active",
        category="evidence_safety",
        severity="info",
        label="Late answer followthrough active",
    ),
    _signal(
        "late_answer_followthrough_blocked_insufficient_evidence",
        category="evidence_safety",
        severity="warning",
        label="Late answer followthrough blocked: insufficient evidence",
        affects_health=True,
    ),
    _signal(
        "answer_anchor_forced_full_resend",
        category="evidence_safety",
        severity="info",
        label="Answer anchor forced full resend",
    ),
    _signal(
        "answer_anchor_verified_current",
        category="evidence_safety",
        severity="info",
        label="Answer anchor verified current",
        release_critical=False,
    ),
    _signal(
        "answer_anchor_delta_allowed",
        category="evidence_safety",
        severity="info",
        label="Answer anchor delta allowed",
        release_critical=False,
    ),
    _signal(
        "answer_ready_exact_evidence_fallback_full_history",
        category="evidence_safety",
        severity="info",
        label="Answer-ready exact evidence: full history fallback",
    ),
    _signal(
        "answer_ready_exact_evidence_in_safe_suffix",
        category="evidence_safety",
        severity="info",
        label="Answer-ready exact evidence in safe suffix",
    ),
    _signal(
        "answer_ready_exact_search_evidence_history_preserved",
        category="evidence_safety",
        severity="info",
        label="Answer-ready search evidence: history preserved",
    ),
    # ---------------------------------------------------------------------------
    # Tool / session event signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "bad_tool_args_event",
        category="session",
        severity="warning",
        label="Bad tool arguments event",
    ),
    _signal(
        "mixed_answer_tool_event",
        category="session",
        severity="info",
        label="Mixed answer+tool event",
        release_critical=False,
    ),
    _signal(
        "toolless_fresh_answer_event",
        category="session",
        severity="info",
        label="Toolless fresh answer event",
        release_critical=False,
    ),
    _signal(
        "unsupported_tool_event",
        category="session",
        severity="warning",
        label="Unsupported tool event",
    ),
    _signal(
        "tool_required_condition_unresolved",
        category="session",
        severity="info",
        label="Tool-required condition unresolved",
    ),
    _signal(
        "tool_required_latch_active",
        category="session",
        severity="info",
        label="Tool-required latch active",
        release_critical=False,
    ),
    _signal(
        "tool_result_order_repair_non_degrading",
        category="session",
        severity="info",
        label="Tool-result order repaired (non-degrading)",
        release_critical=False,
    ),
    _signal(
        "tool_result_cache_hit",
        category="compression",
        severity="info",
        label="Tool-result cache hit",
        release_critical=False,
    ),
    _signal(
        "command_result_cache_hit",
        category="compression",
        severity="info",
        label="Command result cache hit",
        release_critical=False,
    ),
    _signal(
        "command_result_cacheable_seen",
        category="compression",
        severity="info",
        label="Cacheable command result seen",
        release_critical=False,
    ),
    # ---------------------------------------------------------------------------
    # Late-retry answer signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "early_retry_answer_only_satisfied",
        category="recovery",
        severity="info",
        label="Early retry: answer-only satisfied",
        release_critical=False,
    ),
    _signal(
        "early_retry_answer_only_failed_tool",
        category="recovery",
        severity="info",
        label="Early retry: answer-only failed (tool expected)",
        release_critical=False,
    ),
    _signal(
        "early_retry_tool_only_satisfied",
        category="recovery",
        severity="info",
        label="Early retry: tool-only satisfied",
        release_critical=False,
    ),
    _signal(
        "early_retry_tool_only_failed_mixed",
        category="recovery",
        severity="info",
        label="Early retry: tool-only failed (mixed)",
        release_critical=False,
    ),
    _signal(
        "early_retry_tool_only_failed_toolless",
        category="recovery",
        severity="info",
        label="Early retry: tool-only failed (toolless)",
        release_critical=False,
    ),
    _signal(
        "late_retry_answer_only_satisfied",
        category="recovery",
        severity="info",
        label="Late retry: answer-only satisfied",
        release_critical=False,
    ),
    _signal(
        "late_retry_answer_only_failed_tool",
        category="recovery",
        severity="info",
        label="Late retry: answer-only failed (tool expected)",
        release_critical=False,
    ),
    _signal(
        "late_retry_tool_only_satisfied",
        category="recovery",
        severity="info",
        label="Late retry: tool-only satisfied",
        release_critical=False,
    ),
    _signal(
        "late_retry_tool_only_failed_mixed",
        category="recovery",
        severity="info",
        label="Late retry: tool-only failed (mixed)",
        release_critical=False,
    ),
    _signal(
        "late_retry_tool_only_failed_toolless",
        category="recovery",
        severity="info",
        label="Late retry: tool-only failed (toolless)",
        release_critical=False,
    ),
    _signal(
        "retry_prompt_supporting_tool_satisfied",
        category="recovery",
        severity="info",
        label="Retry prompt supporting tool satisfied",
        release_critical=False,
    ),
    _signal(
        "retry_prompt_supporting_tool_missed",
        category="recovery",
        severity="info",
        label="Retry prompt supporting tool missed",
        release_critical=False,
    ),
    _signal(
        "late_answer_followthrough_requested",
        category="recovery",
        severity="info",
        label="Late answer followthrough requested",
        release_critical=False,
    ),
    _signal(
        "late_answer_assembly_repair_answer_only",
        category="recovery",
        severity="info",
        label="Late answer assembly repair: answer-only",
        release_critical=False,
    ),
    _signal(
        "late_answer_assembly_repair_answer_only_requested",
        category="recovery",
        severity="info",
        label="Late answer assembly repair: answer-only requested",
        release_critical=False,
    ),
    _signal(
        "late_answer_assembly_repair_answer_only_resolved",
        category="recovery",
        severity="info",
        label="Late answer assembly repair: answer-only resolved",
        release_critical=False,
    ),
    _signal(
        "late_answer_assembly_repair_answer_only_failed",
        category="recovery",
        severity="info",
        label="Late answer assembly repair: answer-only failed",
        affects_health=True,
    ),
    _signal(
        "late_answer_assembly_repair_tool_only",
        category="recovery",
        severity="info",
        label="Late answer assembly repair: tool-only",
        release_critical=False,
    ),
    # ---------------------------------------------------------------------------
    # State resend / smoothness signals (Phase 0.2.1)
    # ---------------------------------------------------------------------------
    _signal(
        "state_resend_reason_answer_ready_forced_full",
        category="compression",
        severity="info",
        label="State resend: answer-ready forced full",
    ),
    _signal(
        "state_resend_reason_answer_anchor_present_kept_full",
        category="compression",
        severity="info",
        label="State resend: answer anchor kept full",
    ),
    _signal(
        "state_resend_reason_delta_not_smaller",
        category="compression",
        severity="info",
        label="State resend: delta not smaller than full",
        release_critical=False,
    ),
    _signal(
        "state_resend_reason_delta_selected",
        category="compression",
        severity="info",
        label="State resend: delta selected",
        release_critical=False,
    ),
    _signal(
        "state_resend_reason_full_default",
        category="compression",
        severity="info",
        label="State resend: full (default)",
        release_critical=False,
    ),
    _signal(
        "state_resend_reason_history_compression_skipped",
        category="compression",
        severity="info",
        label="State resend: history compression skipped",
        release_critical=False,
    ),
    _signal(
        "state_resend_reason_state_verified_current",
        category="compression",
        severity="info",
        label="State resend: state verified current",
        release_critical=False,
    ),
    _signal(
        "state_resend_reason_tool_compatible_compression_without_resend_change",
        category="compression",
        severity="info",
        label="State resend: tool-compatible compression without resend change",
        release_critical=False,
    ),
    _signal(
        "smoothness_prompt_optimization_active_task",
        category="session",
        severity="info",
        label="Smoothness prompt optimization: active task",
        release_critical=False,
    ),
    _signal(
        "smoothness_history_winnowing_active_loop",
        category="session",
        severity="info",
        label="Smoothness: history winnowing active loop",
        release_critical=False,
    ),
    _signal(
        "smoothness_guarded_history_winnowing_skipped",
        category="session",
        severity="info",
        label="Smoothness: guarded history winnowing skipped",
        release_critical=False,
    ),
    _signal(
        "short_session_history_skipped",
        category="compression",
        severity="info",
        label="Short session: history compression skipped",
        release_critical=False,
    ),
    _signal(
        "semantic_dedup_hit",
        category="compression",
        severity="info",
        label="Semantic deduplication hit",
        release_critical=False,
    ),
    _signal(
        "stable_payload_validation_failed",
        category="provider_safety",
        severity="warning",
        label="Stable payload validation failed",
        affects_health=True,
    ),
    _signal(
        "request_policy_escalation_source_tool_recovery",
        category="recovery",
        severity="info",
        label="Policy escalation source: tool recovery",
        release_critical=False,
    ),
)

SIGNAL_REGISTRY: dict[str, SignalDefinition] = {signal.name: signal for signal in _SIGNALS}

EVIDENCE_SAFETY_SIGNAL_NAMES: tuple[str, ...] = tuple(
    signal.name for signal in _SIGNALS if signal.category == "evidence_safety"
)


def signal_definition(name: str) -> SignalDefinition | None:
    """Return registered metadata for a signal, or None for accepted internal signals."""

    return SIGNAL_REGISTRY.get(name)


def is_registered_signal(name: str) -> bool:
    return name in SIGNAL_REGISTRY


def signals_by_category(category: str) -> tuple[SignalDefinition, ...]:
    return tuple(signal for signal in _SIGNALS if signal.category == category)


def aggregate_signal_category(signals: Mapping[str, int], category: str) -> int:
    names = {signal.name for signal in signals_by_category(category)}
    return sum(int(value) for name, value in signals.items() if name in names)


def aggregate_registered_categories(signals: Mapping[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for name, value in signals.items():
        definition = signal_definition(name)
        if definition is None:
            continue
        totals[definition.category] = totals.get(definition.category, 0) + int(value)
    return totals


def unregistered_signals(signals: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value) for name, value in signals.items() if name not in SIGNAL_REGISTRY}


def debug_unregistered_signals(signals: Mapping[str, int]) -> None:
    """Log unregistered signal keys at session end when TOK_DEBUG=1.

    Intended to be called once per session at teardown. Only emits output
    when the ``TOK_DEBUG`` environment variable is set to ``1``.
    """
    import os

    if os.environ.get("TOK_DEBUG") != "1":
        return
    unknown = unregistered_signals(signals)
    if not unknown:
        return
    names = ", ".join(sorted(unknown))
    logger.debug(
        "Session ended with %d unregistered behavior signal(s): %s",
        len(unknown),
        names,
    )


def warn_unregistered_signals(signals: Mapping[str, int]) -> None:
    """Emit a WARNING for each unregistered signal when TOK_ENFORCE_SIGNAL_REGISTRY=1.

    Intended to be called at session end. No-op when the env var is absent or
    not ``"1"``. This is the enforcement mode that can be made the default in a
    future release once all signals are registered.
    """
    import os

    if os.environ.get("TOK_ENFORCE_SIGNAL_REGISTRY") != "1":
        return
    unknown = unregistered_signals(signals)
    if not unknown:
        return
    names = ", ".join(sorted(unknown))
    logger.warning(
        "TOK_ENFORCE_SIGNAL_REGISTRY: %d unregistered behavior signal(s) emitted: %s",
        len(unknown),
        names,
    )


__all__ = [
    "EVIDENCE_SAFETY_SIGNAL_NAMES",
    "SIGNAL_REGISTRY",
    "SignalDefinition",
    "aggregate_registered_categories",
    "aggregate_signal_category",
    "debug_unregistered_signals",
    "is_registered_signal",
    "signal_definition",
    "signals_by_category",
    "unregistered_signals",
    "warn_unregistered_signals",
]
