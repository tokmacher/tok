"""Preflight helpers for bridge request validation and local recovery."""

from __future__ import annotations

import contextlib
import copy
import json
from typing import Any

from fastapi import Response

from tok.runtime._request_preparation import (
    _restore_latest_assistant_thinking,
    _snapshot_latest_assistant_thinking,
)
from tok.runtime.pipeline.request_validation import (
    bridge_strict_failure_signals,
    canonicalize_anthropic_bridge_body,
    has_blocking_outgoing_failures,
    has_invalid_tool_history_failures,
    has_provider_sensitive_failures,
    has_recoverable_immediate_pairing_failures,
    quarantine_invalid_tool_history_messages,
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_bridge_body,
    validate_anthropic_outgoing_bridge_body,
)
from tok.runtime.smoothness.models import SmoothnessEventType

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
from ._signal_constants import _merge_signal_counts

__all__ = [
    "_count_user_messages_with_mixed_tool_result_content",
    "_count_user_tool_result_split_boundaries",
    "_run_bridge_preflight",
]


def _should_block_invalid_tool_history_locally(
    strict_failures: list[str],
) -> bool:
    return has_invalid_tool_history_failures(strict_failures)


def _should_return_read_burst_hint(failures: list[str]) -> bool:
    return has_provider_sensitive_failures(failures)


def _assistant_tool_use_text_segments(
    content: list[dict[str, Any]],
) -> list[dict[str, list[dict[str, Any]]]]:
    segments: list[dict[str, list[dict[str, Any]]]] = []
    prefix_text: list[dict[str, Any]] = []
    tool_uses: list[dict[str, Any]] = []
    suffix_text: list[dict[str, Any]] = []

    def flush_segment() -> None:
        nonlocal prefix_text, tool_uses, suffix_text
        if not tool_uses:
            prefix_text = []
            suffix_text = []
            return
        segments.append(
            {
                "prefix_text": prefix_text,
                "tool_uses": tool_uses,
                "suffix_text": suffix_text,
            }
        )
        prefix_text = []
        tool_uses = []
        suffix_text = []

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            if tool_uses and suffix_text:
                flush_segment()
            tool_uses.append(copy.deepcopy(block))
            continue
        if block_type in {"text", "thinking", "redacted_thinking"}:
            if tool_uses:
                suffix_text.append(copy.deepcopy(block))
            else:
                prefix_text.append(copy.deepcopy(block))
    flush_segment()
    return segments


def _rewrite_provider_sensitive_large_tool_use_text_interleaving(
    body: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, int]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, False, {}

    rewritten_messages: list[dict[str, Any]] = []
    changed = False
    signals: dict[str, int] = {}
    index = 0

    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            continue

        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role != "assistant" or not isinstance(content, list):
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            continue

        segments = _assistant_tool_use_text_segments(content)
        if len(segments) == 1 and segments[0].get("suffix_text"):
            reordered = copy.deepcopy(message)
            reordered["content"] = segments[0]["prefix_text"] + segments[0]["suffix_text"] + segments[0]["tool_uses"]
            rewritten_messages.append(reordered)
            changed = True
            signals["tok_bridge_tool_use_suffix_text_reordered"] = (
                signals.get("tok_bridge_tool_use_suffix_text_reordered", 0) + 1
            )
            index += 1
            continue
        if len(segments) <= 1:
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            continue

        next_message = messages[index + 1] if index + 1 < len(messages) else None
        if not (
            isinstance(next_message, dict)
            and str(next_message.get("role", "")).strip() == "user"
            and isinstance(next_message.get("content"), list)
        ):
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            continue

        next_content = next_message.get("content")
        assert isinstance(next_content, list), "next_content should be a list"
        user_tool_result_blocks = [
            copy.deepcopy(block)
            for block in next_content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        if len(user_tool_result_blocks) != sum(len(segment["tool_uses"]) for segment in segments):
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            continue
        if len(user_tool_result_blocks) != len(next_content):
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            continue
        expected_ids = [
            str(tool.get("id", "")).strip()
            for segment in segments
            for tool in segment["tool_uses"]
            if isinstance(tool, dict)
        ]
        actual_ids = [str(block.get("tool_use_id", "")).strip() for block in user_tool_result_blocks]
        if expected_ids != actual_ids:
            rewritten_messages.append(copy.deepcopy(message))
            index += 1
            signals["tok_bridge_large_file_read_burst_rewrite_skipped_id_mismatch"] = (
                signals.get("tok_bridge_large_file_read_burst_rewrite_skipped_id_mismatch", 0) + 1
            )
            continue

        changed = True
        signals["tok_bridge_large_file_read_burst_rewritten"] = (
            signals.get("tok_bridge_large_file_read_burst_rewritten", 0) + 1
        )
        result_index = 0
        for segment in segments:
            assistant_message = copy.deepcopy(message)
            assistant_message["content"] = segment["prefix_text"] + segment["suffix_text"] + segment["tool_uses"]
            rewritten_messages.append(assistant_message)
            segment_tool_use_count = len(segment["tool_uses"])
            if segment_tool_use_count:
                user_message = copy.deepcopy(next_message)
                user_message["content"] = user_tool_result_blocks[result_index : result_index + segment_tool_use_count]
                rewritten_messages.append(user_message)
                result_index += segment_tool_use_count
        index += 2
        continue

    rewritten_body = copy.deepcopy(body)
    rewritten_body["messages"] = rewritten_messages
    return rewritten_body, changed, signals


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


def _large_file_read_burst_response(message: str) -> Response:
    return Response(
        content=json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": message,
                    "hint": (
                        "Chunk the file reads into smaller batches, start with "
                        "explore_file/list_large_files, and summarize before "
                        "continuing."
                    ),
                    "recovery": {
                        "strategy": "chunk_large_reads",
                        "max_parallel_file_reads": 2,
                        "step": "emit a compact ReadPlan before the next batch",
                    },
                },
            }
        ),
        status_code=400,
        media_type="application/json",
    )


def _capture_outgoing_guard_forensics(
    session: BridgeSession,
    *,
    event: str,
    body: dict[str, Any],
    original_body: dict[str, Any],
    failures: list[str],
    behavior_signals: dict[str, int],
    payload_source: str | None = None,
) -> None:
    payload_event = {
        "event": event,
        "strict_failures": failures,
        "behavior_signals": behavior_signals,
        "prepared_summary": summarize_message_structure(body.get("messages", [])),
        "prepared_pairing": summarize_bridge_pairing(body.get("messages", [])),
        "provider_safe_summary": summarize_message_structure(original_body.get("messages", [])),
        "provider_safe_pairing": summarize_bridge_pairing(original_body.get("messages", [])),
    }
    if payload_source:
        payload_event["payload_source"] = payload_source
    session.capture_event(payload_event)


def _is_assistant_tool_use_text_interleaving_failure(
    failures: list[str],
) -> bool:
    return "provider_sensitive_assistant_tool_use_text_interleaving" in failures


def _provider_sensitive_payload_source(
    *,
    canonical_body: dict[str, Any],
    original_body: dict[str, Any],
) -> str:
    if _payloads_materially_differ(
        json.dumps(canonical_body).encode(),
        json.dumps(original_body).encode(),
    ):
        return "rewritten"
    return "original"


def _attempt_quarantine_invalid_tool_history(
    body: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, int], list[str]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, False, {}, validate_anthropic_bridge_body(body)
    (
        quarantined_messages,
        changed,
        signals,
    ) = quarantine_invalid_tool_history_messages(messages)
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
    messages: list[dict[str, Any]],
) -> int:
    if not isinstance(messages, list):
        return 0
    mixed_count = 0
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role", "")).strip() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        has_tool_result = any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
        has_non_tool_result = any(isinstance(block, dict) and block.get("type") != "tool_result" for block in content)
        if has_tool_result and has_non_tool_result:
            mixed_count += 1
    return mixed_count


def _count_user_tool_result_split_boundaries(
    messages: list[dict[str, Any]],
) -> int:
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
        if not isinstance(current_content, list) or not isinstance(following_content, list):
            continue
        current_has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in current_content
        )
        current_has_non_tool_result = any(
            isinstance(block, dict) and block.get("type") != "tool_result" for block in current_content
        )
        following_has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in following_content
        )
        if current_has_tool_result and not current_has_non_tool_result and not following_has_tool_result:
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
    _thinking_snapshot = _snapshot_latest_assistant_thinking(original_body.get("messages", []))
    (
        canonical_body,
        bridge_canonicalized,
        bridge_signals,
    ) = canonicalize_anthropic_bridge_body(body)
    if bridge_signals.get("thinking_block_mutated"):
        with contextlib.suppress(Exception):
            session.smoothness_tracker.record(
                SmoothnessEventType.THINKING_BLOCK_MUTATION,
                {"msg_index": bridge_signals.get("thinking_block_mutated_msg_index")},
            )
    _restore_performed = False
    if _thinking_snapshot:
        _restore_performed = _restore_latest_assistant_thinking(canonical_body.get("messages", []), _thinking_snapshot)
    if _restore_performed and bridge_signals.get("thinking_block_mutated"):
        bridge_signals.pop("thinking_block_mutated", None)
        bridge_signals["thinking_block_mutation_restored"] = 1
        logger.debug("thinking_block_mutation_restored | snapshot restore succeeded; penalized signal cleared")
        with contextlib.suppress(Exception):
            session.smoothness_tracker.record(
                SmoothnessEventType.THINKING_BLOCK_MUTATION_RESTORED,
            )
    elif (
        _thinking_snapshot
        and bridge_signals.get("thinking_block_mutated")
        and bridge_signals.get("thinking_block_mutated_has_signature", 0) > 0
    ):
        behavior_signals["tok_bridge_thinking_mutation_unrestored"] = 1
        behavior_signals["tok_release_blocking_thinking_mutation"] = 1
        behavior_signals["tok_bridge_thinking_mutation_degraded_to_provider_safe"] = 1
        mutation_event = _preflight_event_name("bridge_preflight_thinking_mutation_degraded_to_provider_safe", path)
        logger.warning(
            "%s: protected thinking mutation remained after restore attempt; falling back to provider-safe body",
            mutation_event,
        )
        session.capture_event(
            {
                "event": mutation_event,
                "behavior_signals": behavior_signals,
                "thinking_mutation_msg_index": bridge_signals.get("thinking_block_mutated_msg_index"),
                "thinking_mutation_before_hash": bridge_signals.get("thinking_block_mutated_before_hash"),
                "thinking_mutation_after_hash": bridge_signals.get("thinking_block_mutated_after_hash"),
            }
        )
        provider_safe_failures = validate_anthropic_bridge_body(original_body)
        if provider_safe_failures:
            _merge_signal_counts(
                behavior_signals,
                bridge_strict_failure_signals(provider_safe_failures),
            )
            behavior_signals["tok_fallback_activated"] = 1
            _record_fallback_once(session, request_state)
            return (
                canonical_body,
                behavior_signals,
                True,
                _local_bridge_invalid_history_response(
                    "Tok bridge preflight blocked unrecoverable protected-thinking mutation and could not verify provider-safe fallback body."
                ),
            )
        _log_bridge_body_structure(
            mutation_event,
            body=original_body,
            headers=headers,
            original_body=original_body,
            compressed_request=False,
            canonicalized_changed=bridge_canonicalized,
            strict_failures=["thinking_block_mutation_unrestored"],
            reverted_to_original=False,
        )
        return copy.deepcopy(original_body), behavior_signals, True, None
    tool_history_recovery_applied = False
    invalid_tool_history_unrecoverable = False
    strict_failures = validate_anthropic_bridge_body(canonical_body)
    request_fingerprint = _request_fingerprint_diff(headers, canonical_body, original_body)
    should_log_preflight = bool(compressed or bridge_canonicalized or strict_failures)
    _merge_signal_counts(behavior_signals, bridge_signals)
    tool_history_repaired, pairing_repaired = _tool_history_repair_summary(bridge_signals)
    if tool_history_repaired:
        behavior_signals["tok_bridge_tool_history_repaired"] = 1
    if pairing_repaired:
        behavior_signals["tok_bridge_tool_history_pairing_repaired"] = 1
    if (
        request_fingerprint.get("messages_changed")
        and behavior_signals.get("request_policy_reason_structured_tool_loop", 0) > 0
    ):
        with contextlib.suppress(Exception):
            session.smoothness_tracker.record(
                SmoothnessEventType.MESSAGES_CHANGED_OPEN_TOOL_LOOP,
            )
    if (
        request_fingerprint["prompt_caching"]
        and request_fingerprint["body_materially_differs"]
        and (request_fingerprint["messages_changed"] or request_fingerprint["system_changed"])
        and request_fingerprint["cache_topology_changed"]
    ):
        strict_failures = [
            *list(strict_failures),
            "prompt_caching_request_mutated",
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
            behavior_signals["tok_bridge_pairing_degraded_to_provider_safe"] = 1
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
                recovery_signals = session.runtime_session.record_invalid_tool_history_recovery(blocked=False)
                _merge_signal_counts(behavior_signals, recovery_signals)
                if recovery_signals.get("tok_bridge_invalid_tool_history_session_reset", 0):
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
                if emit_ready_log and (should_log_preflight or quarantine_signals):
                    _log_bridge_body_structure(
                        _preflight_event_name("bridge_preflight_repaired_quarantined", path),
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
                            "event": _preflight_event_name("bridge_preflight_repaired_quarantined", path),
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
        if _should_block_invalid_tool_history_locally(strict_failures) or invalid_tool_history_unrecoverable:
            recovery_signals = session.runtime_session.record_invalid_tool_history_recovery(blocked=True)
            _merge_signal_counts(behavior_signals, recovery_signals)
            if recovery_signals.get("tok_bridge_invalid_tool_history_session_reset", 0):
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
                _preflight_event_name("bridge_preflight_rejected_blocked_local", path),
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
                    "event": _preflight_event_name("bridge_preflight_rejected_blocked_local", path),
                    "strict_failures": strict_failures,
                    "behavior_signals": behavior_signals,
                }
            )
            behavior_signals["tok_fallback_activated"] = 1
            _record_fallback_once(session, request_state)
            return (
                canonical_body,
                behavior_signals,
                True,
                _local_bridge_invalid_history_response(
                    "Tok bridge preflight rejected unrepaired tool history before send."
                ),
            )

        if strict_failures:
            _merge_signal_counts(
                behavior_signals,
                bridge_strict_failure_signals(strict_failures),
            )
        if strict_failures and has_blocking_outgoing_failures(strict_failures):
            _log_bridge_body_structure(
                _preflight_event_name("bridge_preflight_rejected_reverted_to_original", path),
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
                tool_history_repaired or pairing_repaired or tool_history_recovery_applied,
                None,
            )

    outgoing_failures = validate_anthropic_outgoing_bridge_body(canonical_body)
    if outgoing_failures:
        _merge_signal_counts(
            behavior_signals,
            bridge_strict_failure_signals(outgoing_failures),
        )
    if outgoing_failures and has_blocking_outgoing_failures(outgoing_failures):
        outgoing_payload_source = _provider_sensitive_payload_source(
            canonical_body=canonical_body,
            original_body=original_body,
        )
        if _should_return_read_burst_hint(outgoing_failures):
            (
                rewritten_body,
                rewritten_changed,
                rewrite_signals,
            ) = _rewrite_provider_sensitive_large_tool_use_text_interleaving(canonical_body)
            if rewritten_changed:
                rewritten_failures = validate_anthropic_outgoing_bridge_body(rewritten_body)
                if not rewritten_failures:
                    behavior_signals.update(rewrite_signals)
                    behavior_signals["tok_bridge_provider_sensitive_degraded_to_provider_safe"] = 1
                    behavior_signals["tok_bridge_prepared_pairing_rejected_local"] = 1
                    behavior_signals["tok_bridge_assistant_tool_use_text_interleaving_blocked"] = 1
                    behavior_signals["request_policy_interleaving_downgrades"] = 1
                    behavior_signals["preflight_block_rewritten_payload"] = 1
                    event_name = _preflight_event_name(
                        "bridge_preflight_large_file_read_burst_rewritten",
                        path,
                    )
                    logger.warning(
                        "%s: rewritten provider-sensitive large file-read burst into provider-safe segments",
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
                        body=rewritten_body,
                        original_body=original_body,
                        failures=outgoing_failures,
                        behavior_signals=behavior_signals,
                        payload_source="rewritten",
                    )
                    _log_bridge_body_structure(
                        event_name,
                        body=rewritten_body,
                        headers=headers,
                        original_body=original_body,
                        compressed_request=False,
                        canonicalized_changed=True,
                        strict_failures=[],
                        reverted_to_original=False,
                    )
                    return (
                        rewritten_body,
                        behavior_signals,
                        True,
                        None,
                    )
        if compressed and _payloads_materially_differ(
            json.dumps(canonical_body).encode(),
            json.dumps(original_body).encode(),
        ):
            degraded_failures = validate_anthropic_outgoing_bridge_body(original_body)
            if not has_blocking_outgoing_failures(degraded_failures):
                interleaving_failure = _is_assistant_tool_use_text_interleaving_failure(outgoing_failures)
                event_name = _preflight_event_name(
                    (
                        "bridge_preflight_assistant_tool_use_text_interleaving_degraded_to_provider_safe"
                        if interleaving_failure
                        else "bridge_preflight_outgoing_degraded_to_provider_safe"
                    ),
                    path,
                )
                if has_provider_sensitive_failures(outgoing_failures):
                    behavior_signals["tok_bridge_provider_sensitive_degraded_to_provider_safe"] = 1
                else:
                    behavior_signals["tok_bridge_pairing_degraded_to_provider_safe"] = 1
                behavior_signals["tok_bridge_prepared_pairing_rejected_local"] = 1
                if interleaving_failure:
                    behavior_signals["tok_bridge_assistant_tool_use_text_interleaving_blocked"] = 1
                    behavior_signals["request_policy_interleaving_downgrades"] = 1
                behavior_signals["preflight_block_rewritten_payload"] = 1
                logger.warning(
                    "%s: prepared request failed final outgoing validation%s; sending provider-safe body (payload_source=rewritten)",
                    event_name,
                    (" due to assistant text interleaved within tool_use batch" if interleaving_failure else ""),
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
                    payload_source="rewritten",
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

        event_name = _preflight_event_name("bridge_preflight_rejected_outgoing_guard_local", path)
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
            "tok_bridge_preflight_rejected_outgoing_guard_local: refusing to send provider-sensitive bridge payload upstream (payload_source=%s)",
            outgoing_payload_source,
        )
        behavior_signals["tok_bridge_preflight_failed_local"] = 1
        behavior_signals["tok_bridge_provider_sensitive_blocked_local"] = 1
        behavior_signals[f"preflight_block_{outgoing_payload_source}_payload"] = 1
        if has_provider_sensitive_failures(outgoing_failures):
            behavior_signals["tok_bridge_provider_pairing_risk_detected"] = 1
        if _is_assistant_tool_use_text_interleaving_failure(outgoing_failures):
            behavior_signals["tok_bridge_assistant_tool_use_text_interleaving_blocked"] = 1
            behavior_signals["request_policy_interleaving_downgrades"] = 1
        session._bump_signals(behavior_signals)
        _capture_outgoing_guard_forensics(
            session,
            event=event_name,
            body=canonical_body,
            original_body=original_body,
            failures=outgoing_failures,
            behavior_signals=behavior_signals,
            payload_source=outgoing_payload_source,
        )
        if _should_return_read_burst_hint(outgoing_failures):
            local_response = _large_file_read_burst_response(
                "Tok bridge preflight rejected a large file-read burst before send."
            )
        else:
            local_response = _local_bridge_invalid_history_response(
                "Tok bridge preflight rejected a provider-sensitive tool concurrency payload before send."
            )
        behavior_signals["tok_fallback_activated"] = 1
        _record_fallback_once(session, request_state)
        return (
            canonical_body,
            behavior_signals,
            True,
            local_response,
        )

    body = canonical_body
    if reset_recovery_state and not tool_history_recovery_applied:
        session.runtime_session.reset_invalid_tool_history_recovery()
    if tool_history_repaired and emit_repair_logs:
        logger.info(
            "%s: repaired historical tool IDs before send",
            _preflight_event_name("bridge_preflight_repaired_tool_history", path),
        )
        session.capture_event(
            {
                "event": _preflight_event_name("bridge_preflight_repaired_tool_history", path),
                "behavior_signals": behavior_signals,
            }
        )
    if pairing_repaired and emit_repair_logs:
        logger.info(
            "%s: repaired tool-result pairing before send",
            _preflight_event_name("bridge_preflight_repaired_tool_result_pairing", path),
        )
        session.capture_event(
            {
                "event": _preflight_event_name("bridge_preflight_repaired_tool_result_pairing", path),
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
        tool_history_repaired or pairing_repaired or tool_history_recovery_applied,
        None,
    )
