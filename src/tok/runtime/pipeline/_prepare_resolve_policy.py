from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tok.runtime.config import (
    _SHORT_SESSION_THRESHOLD,
    TOK_REACQUIRE_STUCK_WINDOW_TURNS,
    TOK_REQUEST_POLICY_STICKY_TURNS,
)
from tok.runtime.core import RuntimeSession
from tok.runtime.policy.semantic_validation import calculate_invisible_pressure
from tok.runtime.types import RuntimeRequest

logger = logging.getLogger("tok.runtime.pipeline._prepare_resolve_policy")


def _messages_contain_tool_material(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return True
        elif message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        return True
    return False


def _resolve_effective_tool_compatible(
    request: RuntimeRequest,
    session: RuntimeSession,
    _translated_messages: list[dict[str, Any]],
    _normalized_tool_events: list[Any],
    behavior_signals: dict[str, int],
) -> tuple[bool, list[str]]:
    if request.request_policy == "forced_baseline" or not request.tool_compatible:
        return False, []
    if request.request_policy == "legacy_tool_compatible":
        return True, ["legacy_default"]

    current_turn = session.bridge_memory.turn
    has_bridge_tool_material = request.adapter_kind == "claude-bridge" and _messages_contain_tool_material(
        request.messages
    )
    if current_turn < _SHORT_SESSION_THRESHOLD and not has_bridge_tool_material:
        behavior_signals["short_session_baseline_mode"] = 1
        return False, ["short_session"]

    recent_stuck_targets = 0
    expired_stuck_targets = 0
    for record in getattr(session, "_hot_summary_records", {}).values():
        stuck_turn = int(getattr(record, "stuck_promotion_turn", 0) or 0)
        if stuck_turn <= 0:
            continue
        if current_turn - stuck_turn <= TOK_REACQUIRE_STUCK_WINDOW_TURNS:
            recent_stuck_targets += 1
        else:
            expired_stuck_targets += 1
    if recent_stuck_targets:
        behavior_signals["structured_tool_loop_stuck_target_recent"] = recent_stuck_targets
    if expired_stuck_targets:
        behavior_signals["structured_tool_loop_stuck_target_expired"] = expired_stuck_targets
    has_stuck_target = recent_stuck_targets > 0
    structured_tool_loop = has_stuck_target or any(
        behavior_signals.get(key, 0) > 0
        for key in (
            "repeated_tool_call",
            "stream_recovery_reacquisition_suppressed",
        )
    )
    reasons: list[str] = []
    if session._request_policy_tool_mode_sticky_turns > 0:
        reasons.append("sticky")
    cooldown_remaining = getattr(session, "_stream_recovery_cooldown_remaining", 0)
    if cooldown_remaining > 0:
        session._stream_recovery_cooldown_remaining = max(0, cooldown_remaining - 1)
    if (
        session._stream_recovery_reacquisition_budget > 0
        or session._stream_recovery_history_floor_budget > 0
        or session._request_policy_stream_recovery_watch_turns > 0
    ):
        reasons.append("stream_recovery")
    if session._request_policy_tool_recovery_watch_turns > 0 or session._invalid_tool_history_recovery_count > 0:
        reasons.append("tool_recovery")
    if any(
        session.pending_behavior_signals.get(key, 0) > 0
        for key in (
            "tok_bridge_provider_sensitive_degraded_to_provider_safe",
            "tok_bridge_provider_sensitive_blocked_local",
            "tok_bridge_provider_pairing_risk_detected",
            "tok_bridge_assistant_tool_use_text_interleaving_blocked",
            "fail_open_retry_upstream_pairing_disagreement",
            "tok_history_pairing_safety_degraded",
        )
    ):
        reasons.append("tool_recovery")
    if structured_tool_loop:
        reasons.append("structured_tool_loop")

    return bool(reasons), reasons


@dataclass
class Step4Result:
    effective_tool_compatible: bool = False
    request_policy_reasons: list[str] = field(default_factory=list)
    request_policy_escalated: bool = False
    behavior_signals: dict[str, int] = field(default_factory=dict)
    mode: Any = None
    policy: Any = None
    current_pressure: int = 0
    hot_hint_metrics: dict[str, int] = field(default_factory=dict)
    saved_tokens: int = 0
    type_breakdown: dict[str, int] = field(default_factory=dict)
    should_skip_history: bool = False
    skip_reason: str = ""
    history_skip_reason: str = ""
    request_policy: str = ""


def run_step_4(
    request: RuntimeRequest,
    session: RuntimeSession,
    translated_messages: list[dict[str, Any]],
    normalized_tool_events: list[Any],
    behavior_signals: dict[str, int],
    should_skip_history: bool,
    skip_reason: str,
    history_skip_reason: str,
    plan_finalization_turn: bool,
) -> Step4Result:
    from tok.runtime.smoothness.models import TokMode

    mode, policy = session.policy_snapshot(request.model)
    saved_tokens = 0
    type_breakdown: dict[str, int] = {}
    hot_hint_metrics: dict[str, int] = {
        "hot_recent_hint_injected": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
        "repeat_tool_collapse_applied": 0,
    }

    _lossless_mode_active = session.current_tok_mode == TokMode.LOSSLESS_TASK_MODE
    if _lossless_mode_active:
        should_skip_history = True
        skip_reason = "lossless_task_mode"
        history_skip_reason = skip_reason
        behavior_signals["lossless_task_mode_history_skipped"] = 1
        logger.info("LOSSLESS_TASK_MODE: skipping active-workset compression entirely")

    current_pressure = calculate_invisible_pressure(behavior_signals)
    request_policy = request.request_policy
    previous_effective_tool_compatible = session._request_policy_last_effective_tool_compatible
    effective_tool_compatible = request.tool_compatible and request_policy == "legacy_tool_compatible"
    request_policy_reasons: list[str] = ["legacy_default"] if effective_tool_compatible else []
    request_policy_escalated = False

    if translated_messages:
        session.bridge_memory.turn += 1
        if request_policy == "natural_first":
            (
                effective_tool_compatible,
                request_policy_reasons,
            ) = _resolve_effective_tool_compatible(
                request,
                session,
                translated_messages,
                normalized_tool_events,
                behavior_signals,
            )
        elif request_policy == "forced_baseline":
            effective_tool_compatible = False
            request_policy_reasons = []

        if plan_finalization_turn:
            behavior_signals["plan_finalization_turn"] = 1
            active_tool_recovery = any(
                reason in {"stream_recovery", "tool_recovery"} for reason in request_policy_reasons
            ) or any(
                session.pending_behavior_signals.get(key, 0) > 0
                for key in (
                    "tok_bridge_provider_sensitive_degraded_to_provider_safe",
                    "tok_bridge_provider_sensitive_blocked_local",
                    "tok_bridge_provider_pairing_risk_detected",
                    "tok_bridge_assistant_tool_use_text_interleaving_blocked",
                    "fail_open_retry_upstream_pairing_disagreement",
                    "tok_history_pairing_safety_degraded",
                )
            )
            if request_policy == "natural_first" and effective_tool_compatible and not active_tool_recovery:
                effective_tool_compatible = False
                request_policy_reasons = ["plan_finalization"]
                behavior_signals["plan_finalization_tool_escalation_suppressed"] = 1

        fresh_tool_mode_trigger = any(
            reason in {"stream_recovery", "tool_recovery", "structured_tool_loop"} for reason in request_policy_reasons
        )
        fresh_tool_mode_trigger = fresh_tool_mode_trigger and (
            session._stream_recovery_reacquisition_budget > 0
            or session._stream_recovery_history_floor_budget > 0
            or session._invalid_tool_history_recovery_count > 0
            or any(
                session.pending_behavior_signals.get(key, 0) > 0
                for key in (
                    "tok_bridge_provider_sensitive_degraded_to_provider_safe",
                    "tok_bridge_provider_sensitive_blocked_local",
                    "tok_bridge_provider_pairing_risk_detected",
                    "tok_bridge_assistant_tool_use_text_interleaving_blocked",
                    "fail_open_retry_upstream_pairing_disagreement",
                )
            )
            or "structured_tool_loop" in request_policy_reasons
        )
        recovery_sticky_continuation = (
            request_policy == "natural_first"
            and previous_effective_tool_compatible
            and effective_tool_compatible
            and not fresh_tool_mode_trigger
        )
        if request_policy == "natural_first" and effective_tool_compatible:
            if not previous_effective_tool_compatible:
                request_policy_escalated = True
                behavior_signals["request_policy_escalations"] = 1
                for reason in request_policy_reasons:
                    if reason in {
                        "stream_recovery",
                        "tool_recovery",
                        "structured_tool_loop",
                    }:
                        behavior_signals[f"request_policy_escalation_source_{reason}"] = 1
            if fresh_tool_mode_trigger:
                session._request_policy_tool_mode_sticky_turns = max(
                    session._request_policy_tool_mode_sticky_turns,
                    TOK_REQUEST_POLICY_STICKY_TURNS,
                )
        elif request_policy == "natural_first" and previous_effective_tool_compatible and not effective_tool_compatible:
            behavior_signals["request_policy_deescalations"] = 1

        if request_policy == "forced_baseline":
            behavior_signals["request_policy_forced_baseline"] = 1
        elif effective_tool_compatible:
            behavior_signals["request_policy_tool_compatible"] = 1
        else:
            behavior_signals["request_policy_natural_first"] = 1
        behavior_signals[f"request_policy_requested_{request_policy}"] = 1
        if request.tool_compatible:
            behavior_signals["request_policy_requested_tool_compatible"] = 1
        else:
            behavior_signals["request_policy_requested_non_tool_compatible"] = 1
        if effective_tool_compatible:
            behavior_signals["request_policy_effective_tool_compatible"] = 1
        else:
            behavior_signals["request_policy_effective_natural_first"] = 1
        if request_policy == "natural_first" and effective_tool_compatible:
            behavior_signals["request_policy_requested_natural_first_effective_tool_compatible"] = 1
        if previous_effective_tool_compatible != effective_tool_compatible:
            if effective_tool_compatible:
                behavior_signals["request_policy_transition_to_tool_compatible"] = 1
            else:
                behavior_signals["request_policy_transition_to_natural_first"] = 1
        else:
            behavior_signals["request_policy_transition_unchanged"] = 1

        for reason in request_policy_reasons:
            behavior_signals[f"request_policy_reason_{reason}"] = 1
        logger.info(
            "request_policy_resolution: requested=%s request_tool_compatible=%s effective_tool_compatible=%s previous_effective_tool_compatible=%s reasons=%s escalated=%s",
            request_policy,
            request.tool_compatible,
            effective_tool_compatible,
            previous_effective_tool_compatible,
            ",".join(request_policy_reasons) if request_policy_reasons else "<none>",
            request_policy_escalated,
        )

        cooldown_suppressed = getattr(session, "_stream_recovery_cooldown_suppressed", False)
        active_recovery_present = (
            session._stream_recovery_reacquisition_budget > 0 or session._stream_recovery_history_floor_budget > 0
        )
        if cooldown_suppressed and not active_recovery_present:
            behavior_signals["request_policy_recovery_cooldown_suppressed"] = 1
        if cooldown_suppressed:
            session._stream_recovery_cooldown_suppressed = False
        if (
            request_policy == "natural_first"
            and session._request_policy_tool_mode_sticky_turns > 0
            and effective_tool_compatible
        ):
            if recovery_sticky_continuation:
                behavior_signals["request_policy_held_by_recovery"] = 1
                behavior_signals["request_policy_recovery_sticky_continuations"] = 1
                session._request_policy_tool_mode_sticky_turns = 0
                session._request_policy_stream_recovery_watch_turns = 0
                session._request_policy_tool_recovery_watch_turns = 0
            else:
                session._request_policy_tool_mode_sticky_turns = max(
                    0, session._request_policy_tool_mode_sticky_turns - 1
                )
        if (
            request_policy == "natural_first"
            and session._request_policy_stream_recovery_watch_turns > 0
            and "stream_recovery" in request_policy_reasons
        ):
            session._request_policy_stream_recovery_watch_turns = max(
                0, session._request_policy_stream_recovery_watch_turns - 1
            )
        if (
            request_policy == "natural_first"
            and session._request_policy_tool_recovery_watch_turns > 0
            and "tool_recovery" in request_policy_reasons
        ):
            session._request_policy_tool_recovery_watch_turns = max(
                0, session._request_policy_tool_recovery_watch_turns - 1
            )
        session._request_policy_last_effective_tool_compatible = effective_tool_compatible
        session._natural_response_acceptable_this_turn = bool(
            request_policy == "natural_first" and request.tool_compatible and not effective_tool_compatible
        )
    else:
        behavior_signals[f"request_policy_requested_{request_policy}"] = 1
        if request.tool_compatible:
            behavior_signals["request_policy_requested_tool_compatible"] = 1
        else:
            behavior_signals["request_policy_requested_non_tool_compatible"] = 1
        if effective_tool_compatible:
            behavior_signals["request_policy_effective_tool_compatible"] = 1
        else:
            behavior_signals["request_policy_effective_natural_first"] = 1
        session._request_policy_last_effective_tool_compatible = effective_tool_compatible
        session._natural_response_acceptable_this_turn = bool(
            request_policy == "natural_first" and request.tool_compatible and not effective_tool_compatible
        )

    return Step4Result(
        effective_tool_compatible=effective_tool_compatible,
        request_policy_reasons=request_policy_reasons,
        request_policy_escalated=request_policy_escalated,
        behavior_signals=behavior_signals,
        mode=mode,
        policy=policy,
        current_pressure=current_pressure,
        hot_hint_metrics=hot_hint_metrics,
        saved_tokens=saved_tokens,
        type_breakdown=type_breakdown,
        should_skip_history=should_skip_history,
        skip_reason=skip_reason,
        history_skip_reason=history_skip_reason,
        request_policy=request_policy,
    )
