from __future__ import annotations

"""Preflight helpers for bridge request validation and local recovery."""

import copy
import json
import random
from typing import Any

from fastapi import Response

from ..runtime.pipeline.request_validation import (
    bridge_strict_failure_signals,
    canonicalize_anthropic_bridge_body,
    has_recoverable_immediate_pairing_failures,
    has_provider_sensitive_failures,
    quarantine_invalid_tool_history_messages,
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_bridge_body,
    validate_anthropic_outgoing_bridge_body,
)
from . import (
    BridgeSession,
    _log_bridge_body_structure,
    _record_fallback_once,
    logger,
)
from ._bridge_comparison import (
    _payloads_materially_differ,
    _request_fingerprint_diff,
)

__all__ = [
    "_compute_rate_limit_backoff_seconds",
    "_count_user_messages_with_mixed_tool_result_content",
    "_count_user_tool_result_split_boundaries",
    "_local_rate_limit_response",
    "_parse_retry_after_seconds",
    "_run_bridge_preflight",
]

_LOCAL_INVALID_TOOL_HISTORY_FAILURES = frozenset(
    {
        "invalid_tool_use_block",
        "invalid_tool_result_block",
        "assistant_tool_use_missing_next_tool_result",
        "assistant_tool_use_incomplete_next_tool_result_coverage",
        "tool_result_unknown_tool_use_id",
        "tool_result_not_immediately_after_assistant_tool_use",
        "user_tool_result_after_text",
        "bridge_wire_model_invalid",
    }
)


def _parse_retry_after_seconds(raw_value: Any) -> float:
    if raw_value is None:
        return 0.0
    try:
        parsed = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed)


def _random_jitter_multiplier() -> float:
    value = random.uniform(0.5, 1.5)
    if isinstance(value, int | float):
        return float(value)
    return 1.0


def _compute_rate_limit_backoff_seconds(
    *,
    attempt: int,
    base_ms: int,
    cap_ms: int,
) -> float:
    bounded_attempt = max(1, attempt)
    bounded_base = max(1, base_ms)
    bounded_cap = max(bounded_base, cap_ms)
    exponential_ms = float(
        min(bounded_cap, bounded_base * (2 ** (bounded_attempt - 1)))
    )
    jitter_multiplier = _random_jitter_multiplier()
    jittered_ms = float(
        min(float(bounded_cap), exponential_ms * jitter_multiplier)
    )
    return max(0.0, jittered_ms / 1000.0)


def _local_rate_limit_response(retry_after_seconds: int) -> Response:
    return Response(
        content=json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Tok bridge local throttling active after repeated upstream 429 responses. Retry later.",
                },
            }
        ),
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": str(max(1, retry_after_seconds))},
    )


def _should_block_invalid_tool_history_locally(
    strict_failures: list[str],
) -> bool:
    return any(
        failure in _LOCAL_INVALID_TOOL_HISTORY_FAILURES
        for failure in strict_failures
    )


def _local_bridge_invalid_history_response(message: str) -> Response:
    return Response(
        content=json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": message,
                },
            }
        ),
        status_code=400,
        media_type="application/json",
    )


def _merge_signal_counts(
    target: dict[str, int], extra: dict[str, int] | None
) -> None:
    if not extra:
        return
    for key, value in extra.items():
        target[key] = target.get(key, 0) + value


def _capture_outgoing_guard_forensics(
    session: BridgeSession,
    *,
    event: str,
    body: dict[str, Any],
    original_body: dict[str, Any],
    failures: list[str],
    behavior_signals: dict[str, int],
) -> None:
    session.capture_event(
        {
            "event": event,
            "strict_failures": failures,
            "behavior_signals": behavior_signals,
            "prepared_summary": summarize_message_structure(
                body.get("messages", [])
            ),
            "prepared_pairing": summarize_bridge_pairing(
                body.get("messages", [])
            ),
            "provider_safe_summary": summarize_message_structure(
                original_body.get("messages", [])
            ),
            "provider_safe_pairing": summarize_bridge_pairing(
                original_body.get("messages", [])
            ),
        }
    )


def _attempt_quarantine_invalid_tool_history(
    body: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, int], list[str]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, False, {}, validate_anthropic_bridge_body(body)
    quarantined_messages, changed, signals = (
        quarantine_invalid_tool_history_messages(messages)
    )
    if not changed:
        return body, False, signals, validate_anthropic_bridge_body(body)
    quarantined_body = copy.deepcopy(body)
    quarantined_body["messages"] = quarantined_messages
    failures = validate_anthropic_bridge_body(quarantined_body)
    return quarantined_body, not failures, signals, failures


def _tool_history_repair_summary(
    bridge_signals: dict[str, int],
) -> tuple[bool, bool]:
    repaired_ids = any(
        bridge_signals.get(key, 0)
        for key in (
            "tok_bridge_tool_id_sanitized",
            "tok_bridge_blank_tool_id_synthesized",
            "tok_bridge_tool_id_deduped",
        )
    )
    pairing_repaired = any(
        bridge_signals.get(key, 0)
        for key in (
            "tok_bridge_tool_result_pairing_repaired",
            "tok_bridge_tool_result_id_rewritten",
            "tok_bridge_tool_result_rewrite_complete",
            "tok_bridge_tool_result_order_repaired",
            "tok_bridge_user_tool_result_text_split",
        )
    )
    return repaired_ids, pairing_repaired


def _count_user_messages_with_mixed_tool_result_content(
    messages: Any,
) -> int:
    if not isinstance(messages, list):
        return 0
    mixed_count = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or str(message.get("role", "")).strip() != "user"
        ):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
        has_non_tool_result = any(
            isinstance(block, dict) and block.get("type") != "tool_result"
            for block in content
        )
        if has_tool_result and has_non_tool_result:
            mixed_count += 1
    return mixed_count


def _count_user_tool_result_split_boundaries(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    boundaries = 0
    for index in range(len(messages) - 1):
        current = messages[index]
        following = messages[index + 1]
        if (
            not isinstance(current, dict)
            or not isinstance(following, dict)
            or str(current.get("role", "")).strip() != "user"
            or str(following.get("role", "")).strip() != "user"
        ):
            continue
        current_content = current.get("content")
        following_content = following.get("content")
        if not isinstance(current_content, list) or not isinstance(
            following_content, list
        ):
            continue
        current_has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in current_content
        )
        current_has_non_tool_result = any(
            isinstance(block, dict) and block.get("type") != "tool_result"
            for block in current_content
        )
        following_has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in following_content
        )
        if (
            current_has_tool_result
            and not current_has_non_tool_result
            and not following_has_tool_result
        ):
            boundaries += 1
    return boundaries


def _preflight_event_name(base_event: str, path: str) -> str:
    if path == "v1/messages/count_tokens":
        return f"{base_event}_count_tokens"
    return base_event


def _run_bridge_preflight(
    session: BridgeSession,
    *,
    body: dict[str, Any],
    original_body: dict[str, Any],
    headers: dict[str, str],
    behavior_signals: dict[str, int],
    compressed: bool,
    request_state: dict[str, bool],
    path: str,
    emit_ready_log: bool = True,
    emit_repair_logs: bool = True,
    reset_recovery_state: bool = True,
) -> tuple[dict[str, Any], dict[str, int], bool, Response | None]:
    """Canonicalize, validate, and recover bridge tool history before send."""
    (
        canonical_body,
        bridge_canonicalized,
        bridge_signals,
    ) = canonicalize_anthropic_bridge_body(body)
    tool_history_recovery_applied = False
    invalid_tool_history_unrecoverable = False
    strict_failures = validate_anthropic_bridge_body(canonical_body)
    request_fingerprint = _request_fingerprint_diff(
        headers, canonical_body, original_body
    )
    should_log_preflight = bool(
        compressed or bridge_canonicalized or strict_failures
    )
    _merge_signal_counts(behavior_signals, bridge_signals)
    tool_history_repaired, pairing_repaired = _tool_history_repair_summary(
        bridge_signals
    )
    if tool_history_repaired:
        behavior_signals["tok_bridge_tool_history_repaired"] = 1
    if pairing_repaired:
        behavior_signals["tok_bridge_tool_history_pairing_repaired"] = 1
    if (
        request_fingerprint["prompt_caching"]
        and request_fingerprint["body_materially_differs"]
        and (
            request_fingerprint["messages_changed"]
            or request_fingerprint["system_changed"]
        )
        and request_fingerprint["cache_topology_changed"]
    ):
        strict_failures = list(strict_failures) + [
            "prompt_caching_request_mutated"
        ]
    _merge_signal_counts(
        behavior_signals,
        bridge_strict_failure_signals(strict_failures),
    )
    if (
        compressed
        and strict_failures
        and has_recoverable_immediate_pairing_failures(strict_failures)
        and _payloads_materially_differ(
            json.dumps(canonical_body).encode(),
            json.dumps(original_body).encode(),
        )
    ):
        degraded_failures = validate_anthropic_bridge_body(original_body)
        if not degraded_failures:
            logger.warning(
                "bridge_preflight_pairing_degraded_to_provider_safe: prepared request violated immediate tool-result pairing; sending provider-safe uncompressed body"
            )
            behavior_signals[
                "tok_bridge_pairing_degraded_to_provider_safe"
            ] = 1
            behavior_signals["tok_bridge_prepared_pairing_rejected_local"] = 1
            session.capture_event(
                {
                    "event": "bridge_preflight_pairing_degraded_to_provider_safe",
                    "strict_failures": strict_failures,
                    "behavior_signals": behavior_signals,
                }
            )
            _log_bridge_body_structure(
                "bridge_preflight_pairing_degraded_to_provider_safe",
                body=original_body,
                headers=headers,
                original_body=original_body,
                compressed_request=False,
                canonicalized_changed=False,
                strict_failures=[],
                reverted_to_original=False,
            )
            return copy.deepcopy(original_body), behavior_signals, True, None
    if strict_failures:
        if _should_block_invalid_tool_history_locally(strict_failures):
            (
                quarantined_body,
                quarantine_valid,
                quarantine_signals,
                quarantine_failures,
            ) = _attempt_quarantine_invalid_tool_history(canonical_body)
            _merge_signal_counts(behavior_signals, quarantine_signals)
            if quarantine_valid:
                recovery_signals = session.runtime_session.record_invalid_tool_history_recovery(
                    blocked=False
                )
                _merge_signal_counts(behavior_signals, recovery_signals)
                if recovery_signals.get(
                    "tok_bridge_invalid_tool_history_session_reset", 0
                ):
                    logger.warning(
                        "bridge_invalid_tool_history_session_reset: cleared hot session state after repeated repair attempts"
                    )
                    session.capture_event(
                        {
                            "event": "bridge_invalid_tool_history_session_reset",
                            "behavior_signals": recovery_signals,
                        }
                    )
                canonical_body = quarantined_body
                bridge_canonicalized = True
                strict_failures = []
                tool_history_recovery_applied = True
                if emit_ready_log and (
                    should_log_preflight or quarantine_signals
                ):
                    _log_bridge_body_structure(
                        _preflight_event_name(
                            "bridge_preflight_repaired_quarantined", path
                        ),
                        body=canonical_body,
                        headers=headers,
                        original_body=original_body,
                        compressed_request=compressed,
                        canonicalized_changed=True,
                        strict_failures=[],
                        reverted_to_original=False,
                    )
                if emit_repair_logs:
                    logger.warning(
                        "tok_bridge_preflight_repaired_quarantined: removed a broken tool exchange and continued with repaired history"
                    )
                    session.capture_event(
                        {
                            "event": _preflight_event_name(
                                "bridge_preflight_repaired_quarantined", path
                            ),
                            "behavior_signals": behavior_signals,
                        }
                    )
            else:
                canonical_body = quarantined_body
                strict_failures = quarantine_failures or strict_failures
                invalid_tool_history_unrecoverable = True
                _merge_signal_counts(
                    behavior_signals,
                    bridge_strict_failure_signals(strict_failures),
                )
        if (
            _should_block_invalid_tool_history_locally(strict_failures)
            or invalid_tool_history_unrecoverable
        ):
            recovery_signals = (
                session.runtime_session.record_invalid_tool_history_recovery(
                    blocked=True
                )
            )
            _merge_signal_counts(behavior_signals, recovery_signals)
            if recovery_signals.get(
                "tok_bridge_invalid_tool_history_session_reset", 0
            ):
                logger.warning(
                    "bridge_invalid_tool_history_session_reset: cleared hot session state after repeated blocked tool-history failures"
                )
                session.capture_event(
                    {
                        "event": "bridge_invalid_tool_history_session_reset",
                        "behavior_signals": recovery_signals,
                    }
                )
            _log_bridge_body_structure(
                _preflight_event_name(
                    "bridge_preflight_rejected_blocked_local", path
                ),
                body=canonical_body,
                headers=headers,
                original_body=original_body,
                compressed_request=compressed,
                canonicalized_changed=bridge_canonicalized,
                strict_failures=strict_failures,
                reverted_to_original=False,
            )
            logger.warning(
                "tok_bridge_preflight_rejected_blocked_local: refusing to send unrepaired invalid tool history upstream"
            )
            behavior_signals["tok_bridge_preflight_failed_local"] = 1
            behavior_signals["tok_bridge_invalid_tool_history_blocked"] = 1
            session._bump_signals(behavior_signals)
            session.capture_event(
                {
                    "event": _preflight_event_name(
                        "bridge_preflight_rejected_blocked_local", path
                    ),
                    "strict_failures": strict_failures,
                    "behavior_signals": behavior_signals,
                }
            )
            return (
                canonical_body,
                behavior_signals,
                True,
                _local_bridge_invalid_history_response(
                    "Tok bridge preflight rejected unrepaired tool history before send."
                ),
            )

        if strict_failures:
            _log_bridge_body_structure(
                _preflight_event_name(
                    "bridge_preflight_rejected_reverted_to_original", path
                ),
                body=canonical_body,
                headers=headers,
                original_body=original_body,
                compressed_request=compressed,
                canonicalized_changed=bridge_canonicalized,
                strict_failures=strict_failures,
                reverted_to_original=True,
            )
            logger.warning(
                "tok_bridge_preflight_rejected_reverted_to_original: reverting rewritten bridge body to original request"
            )
            behavior_signals["tok_bridge_preflight_rejected"] = 1
            behavior_signals["tok_fallback_activated"] = 1
            _record_fallback_once(session, request_state)
            return (
                copy.deepcopy(original_body),
                behavior_signals,
                tool_history_repaired
                or pairing_repaired
                or tool_history_recovery_applied,
                None,
            )

    outgoing_failures = validate_anthropic_outgoing_bridge_body(canonical_body)
    if outgoing_failures:
        _merge_signal_counts(
            behavior_signals,
            bridge_strict_failure_signals(outgoing_failures),
        )
        if compressed and _payloads_materially_differ(
            json.dumps(canonical_body).encode(),
            json.dumps(original_body).encode(),
        ):
            degraded_failures = validate_anthropic_outgoing_bridge_body(
                original_body
            )
            if not degraded_failures:
                event_name = _preflight_event_name(
                    "bridge_preflight_outgoing_degraded_to_provider_safe",
                    path,
                )
                if has_provider_sensitive_failures(outgoing_failures):
                    behavior_signals[
                        "tok_bridge_provider_sensitive_degraded_to_provider_safe"
                    ] = 1
                else:
                    behavior_signals[
                        "tok_bridge_pairing_degraded_to_provider_safe"
                    ] = 1
                behavior_signals[
                    "tok_bridge_prepared_pairing_rejected_local"
                ] = 1
                logger.warning(
                    "%s: prepared request failed final outgoing validation; sending provider-safe body",
                    event_name,
                )
                session.capture_event(
                    {
                        "event": event_name,
                        "strict_failures": outgoing_failures,
                        "behavior_signals": behavior_signals,
                    }
                )
                _capture_outgoing_guard_forensics(
                    session,
                    event=f"{event_name}_forensics",
                    body=canonical_body,
                    original_body=original_body,
                    failures=outgoing_failures,
                    behavior_signals=behavior_signals,
                )
                _log_bridge_body_structure(
                    event_name,
                    body=original_body,
                    headers=headers,
                    original_body=original_body,
                    compressed_request=False,
                    canonicalized_changed=bridge_canonicalized,
                    strict_failures=[],
                    reverted_to_original=False,
                )
                return (
                    copy.deepcopy(original_body),
                    behavior_signals,
                    True,
                    None,
                )

        event_name = _preflight_event_name(
            "bridge_preflight_rejected_outgoing_guard_local", path
        )
        _log_bridge_body_structure(
            event_name,
            body=canonical_body,
            headers=headers,
            original_body=original_body,
            compressed_request=compressed,
            canonicalized_changed=bridge_canonicalized,
            strict_failures=outgoing_failures,
            reverted_to_original=False,
        )
        logger.warning(
            "tok_bridge_preflight_rejected_outgoing_guard_local: refusing to send provider-sensitive bridge payload upstream"
        )
        behavior_signals["tok_bridge_preflight_failed_local"] = 1
        behavior_signals["tok_bridge_provider_sensitive_blocked_local"] = 1
        if has_provider_sensitive_failures(outgoing_failures):
            behavior_signals["tok_bridge_provider_pairing_risk_detected"] = 1
        session._bump_signals(behavior_signals)
        _capture_outgoing_guard_forensics(
            session,
            event=event_name,
            body=canonical_body,
            original_body=original_body,
            failures=outgoing_failures,
            behavior_signals=behavior_signals,
        )
        return (
            canonical_body,
            behavior_signals,
            True,
            _local_bridge_invalid_history_response(
                "Tok bridge preflight rejected a provider-sensitive tool concurrency payload before send."
            ),
        )

    body = canonical_body
    if reset_recovery_state and not tool_history_recovery_applied:
        session.runtime_session.reset_invalid_tool_history_recovery()
    if tool_history_repaired and emit_repair_logs:
        logger.info(
            "%s: repaired historical tool IDs before send",
            _preflight_event_name(
                "bridge_preflight_repaired_tool_history", path
            ),
        )
        session.capture_event(
            {
                "event": _preflight_event_name(
                    "bridge_preflight_repaired_tool_history", path
                ),
                "behavior_signals": behavior_signals,
            }
        )
    if pairing_repaired and emit_repair_logs:
        logger.info(
            "%s: repaired tool-result pairing before send",
            _preflight_event_name(
                "bridge_preflight_repaired_tool_result_pairing", path
            ),
        )
        session.capture_event(
            {
                "event": _preflight_event_name(
                    "bridge_preflight_repaired_tool_result_pairing", path
                ),
                "behavior_signals": behavior_signals,
            }
        )
    if emit_ready_log and should_log_preflight:
        _log_bridge_body_structure(
            _preflight_event_name("bridge_preflight_ready", path),
            body=body,
            headers=headers,
            original_body=original_body,
            compressed_request=compressed,
            canonicalized_changed=bridge_canonicalized,
            strict_failures=[],
            reverted_to_original=False,
        )
    return (
        body,
        behavior_signals,
        tool_history_repaired
        or pairing_repaired
        or tool_history_recovery_applied,
        None,
    )
