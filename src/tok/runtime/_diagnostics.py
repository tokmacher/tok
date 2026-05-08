"""Internal diagnostics snapshot for health endpoint and stats.

Packet 07 (0.1.9): unify data sourcing for health + stats without changing any
public schema or CLI meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiagnosticsSnapshot:
    status: str = "ok"
    bridge: str = "tok"
    port: int = 0
    api_base: str = ""
    mode: str = "unknown"
    request_policy: str = ""
    baseline_only: bool = False
    persistence_failures: int = 0
    fallback_count: int = 0
    actual_tokens: int = 0
    baseline_tokens: int = 0
    session_tokens_saved: int = 0
    baseline_prompt_tokens: int = 0
    prepared_prompt_tokens: int = 0
    saved_prompt_tokens: int = 0
    session_savings_pct: float = 0.0
    session_cost_savings_pct: float = 0.0
    actual_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    semantic_drift_count: int = 0
    fail_open_count: int = 0
    non_tok_count: int = 0
    answer_anchor_miss_count: int = 0
    repeat_search_count: int = 0
    repeat_file_read_count: int = 0
    shell_file_read_normalized_count: int = 0
    shell_file_snapshot_captured_count: int = 0
    repeat_target_hot_count: int = 0
    repeat_target_stuck_count: int = 0
    hot_recent_hint_count: int = 0
    hot_hint_tokens_added: int = 0
    reacquisition_tokens_avoided_estimate: int = 0
    state_resend_full_count: int = 0
    state_resend_delta_count: int = 0
    state_resend_suppressed_count: int = 0
    stream_recovery_attempt_count: int = 0
    stream_recovery_success_text_count: int = 0
    stream_recovery_success_tool_use_count: int = 0
    stream_recovery_fallback_count: int = 0
    stream_recovery_empty_success_count: int = 0
    stream_recovery_read_error_count: int = 0
    tool_history_repaired_count: int = 0
    tool_history_pairing_repaired_count: int = 0
    tool_history_quarantined_count: int = 0
    tool_history_blocked_count: int = 0
    invalid_tool_history_session_reset_count: int = 0
    provider_pairing_disagreement_count: int = 0
    assistant_tool_use_text_interleaving_blocked_count: int = 0
    preflight_block_original_payload_count: int = 0
    preflight_block_rewritten_payload_count: int = 0
    request_policy_natural_first_count: int = 0
    request_policy_tool_compatible_count: int = 0
    request_policy_escalations_count: int = 0
    request_policy_deescalations_count: int = 0
    request_policy_interleaving_downgrades_count: int = 0
    request_policy_reason_stream_recovery_count: int = 0
    request_policy_reason_tool_recovery_count: int = 0
    request_policy_reason_structured_tool_loop_count: int = 0
    request_policy_held_by_recovery_count: int = 0
    session_quality: str = "clean"
    last_degradation_reason: str = ""
    calls: int = 0
    smoothness_score: int = 0
    labour_index: int = 0
    current_mode: str = ""
    stream_instability_events: int = 0
    thinking_mutation_events: int = 0
    task_score: int = 0
    repeated_active_file_reads: int = 0

    def to_health_response(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "bridge": self.bridge,
            "port": self.port,
            "api_base": self.api_base,
            "mode": self.mode,
            "request_policy": self.request_policy,
            "baseline_only": self.baseline_only,
            "persistence_failures": self.persistence_failures,
            "fallback_count": self.fallback_count,
            "actual_tokens": self.actual_tokens,
            "baseline_tokens": self.baseline_tokens,
            "session_tokens_saved": self.session_tokens_saved,
            "baseline_prompt_tokens": self.baseline_prompt_tokens,
            "prepared_prompt_tokens": self.prepared_prompt_tokens,
            "saved_prompt_tokens": self.saved_prompt_tokens,
            "session_savings_pct": self.session_savings_pct,
            "session_cost_savings_pct": self.session_cost_savings_pct,
            "actual_cost_usd": self.actual_cost_usd,
            "baseline_cost_usd": self.baseline_cost_usd,
            "cost_saved_usd": self.cost_saved_usd,
            "semantic_drift_count": self.semantic_drift_count,
            "fail_open_count": self.fail_open_count,
            "non_tok_count": self.non_tok_count,
            "answer_anchor_miss_count": self.answer_anchor_miss_count,
            "repeat_search_count": self.repeat_search_count,
            "repeat_file_read_count": self.repeat_file_read_count,
            "shell_file_read_normalized_count": self.shell_file_read_normalized_count,
            "shell_file_snapshot_captured_count": self.shell_file_snapshot_captured_count,
            "repeat_target_hot_count": self.repeat_target_hot_count,
            "repeat_target_stuck_count": self.repeat_target_stuck_count,
            "hot_recent_hint_count": self.hot_recent_hint_count,
            "hot_hint_tokens_added": self.hot_hint_tokens_added,
            "reacquisition_tokens_avoided_estimate": self.reacquisition_tokens_avoided_estimate,
            "state_resend_full_count": self.state_resend_full_count,
            "state_resend_delta_count": self.state_resend_delta_count,
            "state_resend_suppressed_count": self.state_resend_suppressed_count,
            "stream_recovery_attempt_count": self.stream_recovery_attempt_count,
            "stream_recovery_success_text_count": self.stream_recovery_success_text_count,
            "stream_recovery_success_tool_use_count": self.stream_recovery_success_tool_use_count,
            "stream_recovery_fallback_count": self.stream_recovery_fallback_count,
            "stream_recovery_empty_success_count": self.stream_recovery_empty_success_count,
            "stream_recovery_read_error_count": self.stream_recovery_read_error_count,
            "tool_history_repaired_count": self.tool_history_repaired_count,
            "tool_history_pairing_repaired_count": self.tool_history_pairing_repaired_count,
            "tool_history_quarantined_count": self.tool_history_quarantined_count,
            "tool_history_blocked_count": self.tool_history_blocked_count,
            "invalid_tool_history_session_reset_count": self.invalid_tool_history_session_reset_count,
            "provider_pairing_disagreement_count": self.provider_pairing_disagreement_count,
            "assistant_tool_use_text_interleaving_blocked_count": self.assistant_tool_use_text_interleaving_blocked_count,
            "preflight_block_original_payload_count": self.preflight_block_original_payload_count,
            "preflight_block_rewritten_payload_count": self.preflight_block_rewritten_payload_count,
            "request_policy_natural_first_count": self.request_policy_natural_first_count,
            "request_policy_tool_compatible_count": self.request_policy_tool_compatible_count,
            "request_policy_escalations_count": self.request_policy_escalations_count,
            "request_policy_deescalations_count": self.request_policy_deescalations_count,
            "request_policy_interleaving_downgrades_count": self.request_policy_interleaving_downgrades_count,
            "request_policy_reason_stream_recovery_count": self.request_policy_reason_stream_recovery_count,
            "request_policy_reason_tool_recovery_count": self.request_policy_reason_tool_recovery_count,
            "request_policy_reason_structured_tool_loop_count": self.request_policy_reason_structured_tool_loop_count,
            "request_policy_held_by_recovery_count": self.request_policy_held_by_recovery_count,
            "session_quality": self.session_quality,
            "last_degradation_reason": self.last_degradation_reason,
            "calls": self.calls,
            "smoothness_score": self.smoothness_score,
            "labour_index": self.labour_index,
            "current_mode": self.current_mode,
            "stream_instability_events": self.stream_instability_events,
            "thinking_mutation_events": self.thinking_mutation_events,
            "task_score": self.task_score,
            "repeated_active_file_reads": self.repeated_active_file_reads,
        }

    @classmethod
    def from_session(
        cls,
        *,
        port: int,
        api_base: str,
        request_policy_default: str,
        mode_label: str,
        baseline_only: bool,
        persistence_failures: int,
        session_summary: dict[str, Any],
        signals: dict[str, int],
    ) -> DiagnosticsSnapshot:
        # Signal key names match what the gateway emits; session_summary keys match the tracker.
        fallback_count = max(
            int(session_summary.get("fallback_count", 0)), int(signals.get("tok_fallback_activated", 0))
        )
        return cls(
            port=int(port),
            api_base=str(api_base),
            mode=str(mode_label),
            request_policy=str(request_policy_default),
            baseline_only=bool(baseline_only),
            persistence_failures=int(persistence_failures),
            fallback_count=fallback_count,
            actual_tokens=int(session_summary.get("actual_tokens", 0)),
            baseline_tokens=int(session_summary.get("baseline_tokens", 0)),
            session_tokens_saved=int(session_summary.get("tokens_saved", 0)),
            baseline_prompt_tokens=int(session_summary.get("baseline_prompt_tokens", 0)),
            prepared_prompt_tokens=int(session_summary.get("prepared_prompt_tokens", 0)),
            saved_prompt_tokens=int(session_summary.get("saved_prompt_tokens", 0)),
            session_savings_pct=float(session_summary.get("savings_pct", 0.0)),
            session_cost_savings_pct=float(
                session_summary.get("cost_savings_pct", session_summary.get("savings_pct", 0.0))
            ),
            actual_cost_usd=float(session_summary.get("actual_cost_usd", 0.0)),
            baseline_cost_usd=float(session_summary.get("baseline_cost_usd", 0.0)),
            cost_saved_usd=float(session_summary.get("cost_saved_usd", 0.0)),
            semantic_drift_count=int(
                session_summary.get("semantic_drift_count", signals.get("semantic_drift_detected", 0))
            ),
            fail_open_count=int(session_summary.get("fail_open_count", signals.get("fail_open_compat_response", 0))),
            non_tok_count=int(session_summary.get("non_tok_count", signals.get("non_tok_response", 0))),
            answer_anchor_miss_count=int(session_summary.get("answer_anchor_miss_count", 0)),
            repeat_search_count=int(signals.get("repeat_search", 0)),
            repeat_file_read_count=int(signals.get("repeat_file_read", 0)),
            shell_file_read_normalized_count=int(signals.get("shell_file_read_normalized", 0)),
            shell_file_snapshot_captured_count=int(signals.get("shell_file_snapshot_captured", 0)),
            repeat_target_hot_count=int(signals.get("repeat_target_hot", 0)),
            repeat_target_stuck_count=int(signals.get("repeat_target_stuck", 0)),
            hot_recent_hint_count=int(signals.get("hot_recent_hint_injected", 0)),
            hot_hint_tokens_added=int(
                session_summary.get("hot_hint_tokens_added", signals.get("hot_hint_tokens_added", 0))
            ),
            reacquisition_tokens_avoided_estimate=int(
                session_summary.get(
                    "reacquisition_tokens_avoided_estimate",
                    signals.get("reacquisition_tokens_avoided_estimate", 0),
                )
            ),
            state_resend_full_count=int(signals.get("state_resend_full_turn", 0)),
            state_resend_delta_count=int(signals.get("state_resend_delta_turn", 0)),
            state_resend_suppressed_count=int(signals.get("state_resend_suppressed_turn", 0)),
            stream_recovery_attempt_count=max(
                int(session_summary.get("stream_recovery_attempt_count", 0)),
                int(signals.get("stream_recovery_started", 0)),
            ),
            stream_recovery_success_text_count=int(
                session_summary.get(
                    "stream_recovery_success_text_count", signals.get("stream_recovery_success_text", 0)
                )
            ),
            stream_recovery_success_tool_use_count=int(
                session_summary.get(
                    "stream_recovery_success_tool_use_count",
                    signals.get("stream_recovery_success_tool_use", 0),
                )
            ),
            stream_recovery_fallback_count=int(
                session_summary.get("stream_recovery_fallback_count", signals.get("stream_recovery_fallback", 0))
            ),
            stream_recovery_empty_success_count=int(
                session_summary.get(
                    "stream_recovery_empty_success_count",
                    signals.get("stream_recovery_empty_success", 0),
                )
            ),
            stream_recovery_read_error_count=int(
                session_summary.get(
                    "stream_recovery_read_error_count",
                    signals.get("stream_recovery_read_error", 0),
                )
            ),
            tool_history_repaired_count=int(session_summary.get("tool_history_repaired_count", 0)),
            tool_history_pairing_repaired_count=int(session_summary.get("tool_history_pairing_repaired_count", 0)),
            tool_history_quarantined_count=int(session_summary.get("tool_history_quarantined_count", 0)),
            tool_history_blocked_count=max(
                int(session_summary.get("tool_history_blocked_count", 0)),
                int(signals.get("tok_bridge_invalid_tool_history_blocked", 0)),
            ),
            invalid_tool_history_session_reset_count=int(
                session_summary.get("invalid_tool_history_session_reset_count", 0)
            ),
            provider_pairing_disagreement_count=int(session_summary.get("provider_pairing_disagreement_count", 0)),
            assistant_tool_use_text_interleaving_blocked_count=int(
                session_summary.get(
                    "assistant_tool_use_text_interleaving_blocked_count",
                    signals.get("tok_bridge_assistant_tool_use_text_interleaving_blocked", 0),
                )
            ),
            preflight_block_original_payload_count=int(
                session_summary.get(
                    "preflight_block_original_payload_count",
                    signals.get("preflight_block_original_payload", 0),
                )
            ),
            preflight_block_rewritten_payload_count=int(
                session_summary.get(
                    "preflight_block_rewritten_payload_count",
                    signals.get("preflight_block_rewritten_payload", 0),
                )
            ),
            request_policy_natural_first_count=int(session_summary.get("request_policy_natural_first_count", 0)),
            request_policy_tool_compatible_count=int(session_summary.get("request_policy_tool_compatible_count", 0)),
            request_policy_escalations_count=int(session_summary.get("request_policy_escalations_count", 0)),
            request_policy_deescalations_count=int(session_summary.get("request_policy_deescalations_count", 0)),
            request_policy_interleaving_downgrades_count=int(
                session_summary.get("request_policy_interleaving_downgrades_count", 0)
            ),
            request_policy_reason_stream_recovery_count=int(
                session_summary.get(
                    "request_policy_reason_stream_recovery_count",
                    signals.get("request_policy_reason_stream_recovery", 0),
                )
            ),
            request_policy_reason_tool_recovery_count=int(
                session_summary.get(
                    "request_policy_reason_tool_recovery_count",
                    signals.get("request_policy_reason_tool_recovery", 0),
                )
            ),
            request_policy_reason_structured_tool_loop_count=int(
                session_summary.get(
                    "request_policy_reason_structured_tool_loop_count",
                    signals.get("request_policy_reason_structured_tool_loop", 0),
                )
            ),
            request_policy_held_by_recovery_count=int(
                session_summary.get(
                    "request_policy_held_by_recovery_count",
                    signals.get("request_policy_held_by_recovery", 0)
                    + signals.get("request_policy_recovery_sticky_continuations", 0),
                )
            ),
            session_quality=str(session_summary.get("session_quality", "clean")),
            last_degradation_reason=str(session_summary.get("last_degradation_reason", "")),
            calls=int(session_summary.get("calls", 0)),
            smoothness_score=int(session_summary.get("smoothness_score", 0)),
            labour_index=int(session_summary.get("labour_index", 0)),
            current_mode=str(session_summary.get("current_mode", "")),
            stream_instability_events=int(signals.get("stream_instability_events", 0)),
            thinking_mutation_events=int(signals.get("thinking_mutation_events", 0)),
            task_score=int(session_summary.get("task_score", 0)),
            repeated_active_file_reads=int(signals.get("repeat_file_read", 0)),
        )

    @classmethod
    def from_health_response(cls, payload: dict[str, Any]) -> DiagnosticsSnapshot:
        data: dict[str, Any] = {}
        for name, field in cls.__dataclass_fields__.items():
            val = payload.get(name, field.default)
            ftype = field.type
            if isinstance(ftype, str):
                ftype_name = ftype
            else:
                ftype_name = getattr(ftype, "__name__", "")
            if ftype_name == "int":
                data[name] = int(val) if val is not None else 0
            elif ftype_name == "float":
                data[name] = float(val) if val is not None else 0.0
            elif ftype_name == "bool":
                if val is None:
                    data[name] = False
                elif isinstance(val, bool):
                    data[name] = val
                else:
                    data[name] = bool(val)
            else:
                data[name] = str(val) if val is not None else ""
        return cls(**data)
