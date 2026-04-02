"""Quality heuristics for savings reporting."""

from __future__ import annotations

SESSION_STATS_FILENAME = "tok_savings.tok"
GLOBAL_LEDGER_FILENAME = "global_savings.tok"
BASELINE_ONLY_SIGNAL = "baseline_only_session"
FALLBACK_SIGNAL = "tok_fallback_activated"
PROMPT_METRIC_KEYS = (
    "baseline_prompt_tokens",
    "prepared_prompt_tokens",
    "saved_prompt_tokens",
    "hot_hint_tokens_added",
    "reacquisition_tokens_avoided_estimate",
)


def degradation_reason(signals: dict[str, int], *, baseline_only: bool) -> str:
    stream_transport_count = (
        int(signals.get("stream_recovery_read_error", 0))
        + int(signals.get("stream_recovery_empty_success", 0))
        + int(signals.get("stream_recovery_retry", 0))
        + int(signals.get("stream_recovery_fallback", 0))
    )
    request_shape_count = (
        int(signals.get("preflight_block_original_payload", 0))
        + int(signals.get("preflight_block_rewritten_payload", 0))
        + int(signals.get("tok_bridge_provider_pairing_risk_detected", 0))
        + int(
            signals.get(
                "tok_bridge_assistant_tool_use_text_interleaving_blocked",
                0,
            )
        )
        + int(signals.get("fail_open_retry_upstream_pairing_disagreement", 0))
    )
    recovery_holdover_count = int(
        signals.get("request_policy_held_by_recovery", 0)
    ) + int(signals.get("request_policy_recovery_sticky_continuations", 0))
    tool_history_recovery_count = (
        int(signals.get("tok_bridge_tool_history_repaired", 0))
        + int(signals.get("tok_bridge_tool_history_pairing_repaired", 0))
        + int(signals.get("tok_bridge_invalid_tool_history_quarantined", 0))
        + int(signals.get("tok_bridge_invalid_tool_history_blocked", 0))
    )
    if baseline_only or signals.get(BASELINE_ONLY_SIGNAL, 0):
        return "baseline fallback"
    if (
        request_shape_count >= stream_transport_count
        and request_shape_count >= recovery_holdover_count
        and request_shape_count > 0
    ):
        return "request-shape incompatibility"
    if (
        stream_transport_count >= recovery_holdover_count
        and stream_transport_count > 0
    ):
        return "stream transport instability"
    if recovery_holdover_count > 0:
        return "recovery holdover"
    if tool_history_recovery_count > 0:
        return "heavy tool-mode recovery"
    if signals.get("fail_open_compat_response", 0) or signals.get(
        "processing_error", 0
    ):
        return "fail-open compatibility"
    if signals.get("semantic_drift_detected", 0) or signals.get(
        "non_tok_response", 0
    ):
        return "response contract drift"
    if signals.get("repeat_file_read", 0) or signals.get("repeat_search", 0):
        return "context reacquisition"
    if signals.get("answer_anchor_present", 0) == 0 and (
        signals.get("state_resend_suppressed_turn", 0)
        or signals.get("state_resend_delta_turn", 0)
        or signals.get("state_resend_full_turn", 0)
    ):
        return "answer anchor retention"
    return ""


def session_quality(
    signals: dict[str, int],
    *,
    baseline_only: bool,
    tokens_saved: int = 0,
) -> str:
    if baseline_only:
        return "degraded"
    if signals.get("tok_bridge_invalid_tool_history_session_reset", 0):
        return "degraded"
    if (
        signals.get(FALLBACK_SIGNAL, 0)
        or signals.get("semantic_drift_detected", 0)
        or signals.get("fail_open_compat_response", 0)
        or signals.get("fail_open_retry_upstream_pairing_disagreement", 0)
        or signals.get("tok_bridge_provider_pairing_risk_detected", 0)
        or signals.get(
            "tok_bridge_assistant_tool_use_text_interleaving_blocked", 0
        )
        or signals.get("tok_bridge_tool_history_repaired", 0)
        or signals.get("tok_bridge_tool_history_pairing_repaired", 0)
        or signals.get("stream_recovery_retry", 0)
        or signals.get("stream_recovery_fallback", 0)
        or signals.get("tok_bridge_invalid_tool_history_quarantined", 0)
        or signals.get("tok_bridge_invalid_tool_history_blocked", 0)
        or signals.get("repeat_file_read", 0)
        or signals.get("repeat_search", 0)
        or (
            tokens_saved > 0
            and signals.get("answer_anchor_present", 0) == 0
            and (
                signals.get("state_resend_suppressed_turn", 0)
                or signals.get("state_resend_delta_turn", 0)
                or signals.get("state_resend_full_turn", 0)
            )
        )
    ):
        return "watch"
    return "clean"
