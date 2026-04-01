from __future__ import annotations

"""Gateway app-construction helpers behind the public interface."""

import copy
import asyncio
import hashlib
import json
import os
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import httpcore
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from ..runtime.pipeline.request_validation import (
    bridge_strict_failure_signals,
    canonicalize_anthropic_bridge_body,
    has_recoverable_immediate_pairing_failures,
    normalize_tool_use_blocks,
    quarantine_invalid_tool_history_messages,
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_bridge_body,
)
from ..runtime.policy.translator import IS_TOK
from ..universal_runtime import RuntimeRequest
from . import (
    ANTHROPIC_API_BASE,
    BridgeSession,
    _RUNTIME,
    _log_bridge_body_structure,
    _materialize_stream_tool_blocks,
    _record_fallback_once,
    _response_contract_for_mode,
    logger,
)

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

_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT: int = int(
    os.getenv("TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT", "2")
)


def _parse_retry_after_seconds(raw_value: Any) -> float:
    if raw_value is None:
        return 0.0
    try:
        parsed = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed)


def _compute_rate_limit_backoff_seconds(
    *,
    attempt: int,
    base_ms: int,
    cap_ms: int,
) -> float:
    bounded_attempt = max(1, attempt)
    bounded_base = max(1, base_ms)
    bounded_cap = max(bounded_base, cap_ms)
    exponential_ms = min(
        bounded_cap, bounded_base * (2 ** (bounded_attempt - 1))
    )
    jitter_multiplier = random.uniform(0.5, 1.5)
    jittered_ms = min(bounded_cap, exponential_ms * jitter_multiplier)
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


def _tool_use_only_signature(blocks: list[dict[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        normalized.append(
            {
                "name": str(block.get("name", "")),
                "input": block.get("input", {}),
            }
        )
    if not normalized:
        return ""
    payload = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


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
            behavior_signals[
                "tok_bridge_prepared_pairing_rejected_local"
            ] = 1
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
                recovery_signals = (
                    session.runtime_session.record_invalid_tool_history_recovery(
                        blocked=False
                    )
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
        if _should_block_invalid_tool_history_locally(
            strict_failures
        ) or invalid_tool_history_unrecoverable:
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

from ._bridge_comparison import (
    _payloads_materially_differ,
    _request_fingerprint_diff,
    _safe_headers,
)


async def buffer_strip_restream_impl(
    session: BridgeSession,
    client: httpx.AsyncClient,
    response: httpx.Response,
    input_saved_tokens: int = 0,
    type_breakdown: dict[str, int] | None = None,
    behavior_signals: dict[str, int] | None = None,
    prompt_metrics: dict[str, int] | None = None,
    tool_compatible: bool = False,
    request_method: str = "POST",
    request_url: str = "",
    request_headers: dict[str, str] | None = None,
    request_content: bytes | None = None,
    request_state: dict[str, bool] | None = None,
) -> AsyncIterator[bytes]:
    """Buffer the full SSE stream, translate Tok -> readable English/tool_use, re-emit."""
    try:
        raw = b""
        read_error: str | None = None
        try:
            async for chunk in response.aiter_bytes():
                raw += chunk
        except (httpx.ReadError, httpcore.ReadError) as e:
            read_error = str(e)
            logger.warning(
                "Stream read error during buffering, emitting partial content: %s",
                read_error,
            )
        text = raw.decode("utf-8", errors="replace")
        sse_events = text.split("\n\n")

        accumulated: list[str] = []
        sse_model: str = "unknown"
        sse_usage: dict[str, Any] = {}
        stream_blocks: dict[int, dict[str, Any]] = {}
        stream_order: list[int] = []

        for event_str in sse_events:
            for line in event_str.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    d = json.loads(line[6:])
                    etype = d.get("type", "")
                    if etype == "message_start":
                        msg = d.get("message", {})
                        sse_model = msg.get("model", sse_model)
                        sse_usage = msg.get("usage", sse_usage)
                    elif etype == "content_block_start":
                        index = d.get("index")
                        block = d.get("content_block", {})
                        if not isinstance(index, int) or not isinstance(
                            block, dict
                        ):
                            continue
                        if index not in stream_blocks:
                            stream_order.append(index)
                        block_type = block.get("type", "text")
                        if block_type == "tool_use":
                            stream_blocks[index] = {
                                "type": "tool_use",
                                "id": block.get("id", ""),
                                "name": block.get("name", "unknown"),
                                "input": dict(block.get("input", {}))
                                if isinstance(block.get("input", {}), dict)
                                else {},
                                "partial_json": [],
                            }
                        else:
                            stream_blocks[index] = {
                                "type": "text",
                                "text": str(block.get("text", "")),
                            }
                    elif etype == "message_delta":
                        sse_usage.update(d.get("usage", {}))
                    elif etype == "content_block_delta":
                        index = d.get("index")
                        delta = d.get("delta", {})
                        delta_type = delta.get("type", "")
                        if not isinstance(index, int):
                            continue
                        block = stream_blocks.setdefault(
                            index, {"type": "text", "text": ""}
                        )
                        if index not in stream_order:
                            stream_order.append(index)
                        if delta_type == "text_delta":
                            piece = delta.get("text", "")
                            accumulated.append(piece)
                            block["type"] = "text"
                            block["text"] = str(block.get("text", "")) + piece
                        elif delta_type == "input_json_delta":
                            block["type"] = "tool_use"
                            partials = block.setdefault("partial_json", [])
                            if isinstance(partials, list):
                                partials.append(delta.get("partial_json", ""))
                        logger.debug(
                            "Delta type: %s, partial: %s",
                            delta_type,
                            str(delta)[:50],
                        )
                except (json.JSONDecodeError, KeyError):
                    pass

        full_text = "".join(accumulated)
        logger.info(
            "SSE parsed: %d events, accumulated %d chars",
            len(sse_events),
            len(full_text),
        )
        if full_text:
            logger.info("Raw text sample: %s", full_text[:200])

        stream_behavior_signals = dict(behavior_signals or {})
        if read_error:
            stream_behavior_signals["stream_buffer_read_error"] = 1

        tool_blocks = _materialize_stream_tool_blocks(
            stream_blocks, stream_order
        )
        content_blocks = _response_contract_for_mode(
            full_text, tool_compatible=tool_compatible
        ).content_blocks
        translated_blocks = content_blocks + tool_blocks
        has_visible_blocks = any(
            block.get("type") == "tool_use"
            or (
                block.get("type") == "text"
                and str(block.get("text", "")).strip()
            )
            for block in translated_blocks
        )
        logger.info(
            "Translated %d content blocks from %d chars",
            len(translated_blocks),
            len(full_text),
        )
        if full_text:
            logger.info(
                "Response contains Tok markers: %s",
                bool(IS_TOK.search(full_text)),
            )
            logger.info("Response tool_compatible mode: %s", tool_compatible)
            logger.info("Response text sample: %s", full_text[:200])

            processed = _RUNTIME.process_response(
                full_text,
                model=sse_model if sse_model != "unknown" else "",
                session=session.runtime_session,
                behavior_signals=stream_behavior_signals or None,
                tool_compatible=tool_compatible,
            )
            content_blocks = processed.content_blocks + tool_blocks
            response_signals = processed.behavior_signals

            logger.info("Response mode: %s", processed.mode)
            logger.info("Response signals: %s", response_signals)
            logger.info("Content blocks count: %d", len(content_blocks))
            translated_blocks = content_blocks
            has_visible_blocks = any(
                block.get("type") == "tool_use"
                or (
                    block.get("type") == "text"
                    and str(block.get("text", "")).strip()
                )
                for block in translated_blocks
            )
        recovery_required = not has_visible_blocks and (
            read_error is not None or len(translated_blocks) == 0
        )
        if recovery_required:
            stream_behavior_signals["stream_empty_after_success"] = 1
            session.runtime_session._stream_recovery_reacquisition_budget = 1
            session.runtime_session._stream_recovery_history_floor_budget = 1
            recovered = False
            recovery_model = ""
            recovery_usage: dict[str, Any] = {}
            if request_content and request_url:
                stream_behavior_signals["stream_recovery_started"] = 1
                stream_behavior_signals["stream_recovery_retry"] = 1
                logger.warning(
                    "stream_recovery_retry_started: empty streamed success detected; retrying upstream non-stream"
                )
                recovery_payload = request_content
                try:
                    parsed_request = json.loads(request_content)
                    if isinstance(parsed_request, dict):
                        parsed_request = dict(parsed_request)
                        parsed_request["stream"] = False
                        recovery_payload = json.dumps(parsed_request).encode()
                except Exception:
                    recovery_payload = request_content
                async with httpx.AsyncClient(timeout=300.0) as retry_client:
                    retry_request = retry_client.build_request(
                        request_method,
                        request_url,
                        headers=request_headers or {},
                        content=recovery_payload,
                    )
                    retry_response = await retry_client.send(
                        retry_request, stream=False
                    )
                    if retry_response.status_code == 200:
                        try:
                            retry_json = retry_response.json()
                        except Exception:
                            retry_json = None
                        if isinstance(retry_json, dict):
                            recovery_model = str(retry_json.get("model", ""))
                            recovery_usage = retry_json.get("usage", {})
                            passthrough_blocks = [
                                block
                                for block in retry_json.get("content", [])
                                if isinstance(block, dict)
                                and block.get("type") != "text"
                            ]
                            passthrough_blocks, passthrough_signals = (
                                normalize_tool_use_blocks(
                                    passthrough_blocks,
                                    seed_prefix="toolu_recovery",
                                )
                            )
                            _merge_signal_counts(
                                stream_behavior_signals, passthrough_signals
                            )
                            retry_text = "".join(
                                str(block.get("text", ""))
                                for block in retry_json.get("content", [])
                                if isinstance(block, dict)
                                and block.get("type") == "text"
                            )
                            retry_output_saved = 0
                            if retry_text:
                                retry_processed = _RUNTIME.process_response(
                                    retry_text,
                                    model=str(retry_json.get("model", "")),
                                    session=session.runtime_session,
                                    behavior_signals=stream_behavior_signals
                                    or None,
                                    tool_compatible=tool_compatible,
                                )
                                response_signals = (
                                    retry_processed.behavior_signals
                                )
                                translated_blocks = (
                                    retry_processed.content_blocks
                                    + passthrough_blocks
                                )
                                retry_output_saved = (
                                    retry_processed.output_saved_tokens
                                )
                            elif passthrough_blocks:
                                retry_processed = _RUNTIME.process_response(
                                    "",
                                    model=str(retry_json.get("model", "")),
                                    session=session.runtime_session,
                                    behavior_signals=stream_behavior_signals
                                    or None,
                                    tool_compatible=tool_compatible,
                                )
                                response_signals = (
                                    retry_processed.behavior_signals
                                )
                                translated_blocks = passthrough_blocks
                            else:
                                response_signals = dict(
                                    stream_behavior_signals
                                )
                                translated_blocks = []
                            recovered = any(
                                block.get("type") == "tool_use"
                                or (
                                    block.get("type") == "text"
                                    and str(block.get("text", "")).strip()
                                )
                                for block in translated_blocks
                            )
                            if recovered:
                                recovered_text = any(
                                    block.get("type") == "text"
                                    and str(block.get("text", "")).strip()
                                    for block in translated_blocks
                                )
                                recovered_tool_use = any(
                                    block.get("type") == "tool_use"
                                    for block in translated_blocks
                                )
                                if recovered_tool_use and not recovered_text:
                                    signature = _tool_use_only_signature(
                                        translated_blocks
                                    )
                                    if (
                                        signature
                                        and signature
                                        == session.runtime_session._stream_recovery_tool_use_only_signature
                                    ):
                                        session.runtime_session._stream_recovery_tool_use_only_repeat_count += 1
                                    else:
                                        session.runtime_session._stream_recovery_tool_use_only_signature = signature
                                        session.runtime_session._stream_recovery_tool_use_only_repeat_count = 1

                                    if (
                                        session.runtime_session._stream_recovery_tool_use_only_repeat_count
                                        >= _STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT
                                    ):
                                        stream_behavior_signals[
                                            "stream_recovery_loop_broken"
                                        ] = 1
                                        stream_behavior_signals[
                                            "stream_recovery_fallback"
                                        ] = 1
                                        logger.warning(
                                            "stream_recovery_loop_breaker_triggered: repeated identical tool_use-only recovery detected; falling back to avoid retry loop"
                                        )
                                        recovered = False
                                else:
                                    session.runtime_session._stream_recovery_tool_use_only_signature = ""
                                    session.runtime_session._stream_recovery_tool_use_only_repeat_count = 0

                            if recovered:
                                recovery_success_signals: dict[str, int] = {}
                                if recovered_text:
                                    stream_behavior_signals[
                                        "stream_recovery_success_text"
                                    ] = 1
                                    recovery_success_signals[
                                        "stream_recovery_success_text"
                                    ] = 1
                                    logger.info(
                                        "stream_recovery_succeeded_text: recovered empty streamed success via non-stream retry"
                                    )
                                if recovered_tool_use:
                                    stream_behavior_signals[
                                        "stream_recovery_success_tool_use"
                                    ] = 1
                                    recovery_success_signals[
                                        "stream_recovery_success_tool_use"
                                    ] = 1
                                    logger.info(
                                        "stream_recovery_succeeded_tool_use: recovered empty streamed success via non-stream retry"
                                    )
                                response_signals = dict(
                                    response_signals or {}
                                )
                                _merge_signal_counts(
                                    response_signals,
                                    recovery_success_signals,
                                )
                                retry_usage = retry_json.get("usage", {})
                                retry_model = str(
                                    retry_json.get("model", "")
                                )
                                if retry_model and retry_usage:
                                    session.tracker.record_call(
                                        model=retry_model,
                                        actual_input=retry_usage.get(
                                            "input_tokens", 0
                                        ),
                                        actual_output=retry_usage.get(
                                            "output_tokens", 0
                                        ),
                                        cache_read=retry_usage.get(
                                            "cache_read_input_tokens", 0
                                        ),
                                        cache_write=retry_usage.get(
                                            "cache_creation_input_tokens", 0
                                        ),
                                        input_saved=input_saved_tokens,
                                        output_saved=retry_output_saved,
                                        type_breakdown=type_breakdown,
                                        behavior_signals=response_signals
                                        or None,
                                        prompt_metrics=prompt_metrics,
                                    )
                                message_start = {
                                    "type": "message_start",
                                    "message": {
                                        "model": retry_model or sse_model,
                                        "usage": retry_usage,
                                    },
                                }
                                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()
                                for i, block in enumerate(translated_blocks):
                                    if block.get("type") == "text":
                                        start = {
                                            "type": "content_block_start",
                                            "index": i,
                                            "content_block": {
                                                "type": "text",
                                                "text": "",
                                            },
                                        }
                                        delta = {
                                            "type": "content_block_delta",
                                            "index": i,
                                            "delta": {
                                                "type": "text_delta",
                                                "text": block.get("text", ""),
                                            },
                                        }
                                    else:
                                        start = {
                                            "type": "content_block_start",
                                            "index": i,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": block.get("id", ""),
                                                "name": block.get(
                                                    "name", "unknown"
                                                ),
                                                "input": {},
                                            },
                                        }
                                        delta = {
                                            "type": "content_block_delta",
                                            "index": i,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": json.dumps(
                                                    block.get("input", {})
                                                ),
                                            },
                                        }
                                    stop = {
                                        "type": "content_block_stop",
                                        "index": i,
                                    }
                                    yield f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode()
                                    yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode()
                                    yield f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode()
                                yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
                                return
            if not recovered:
                stream_behavior_signals["stream_recovery_fallback"] = 1
                logger.warning(
                    "stream_recovery_fallback: non-stream retry produced no visible content; recording fallback"
                )
                if recovery_model and recovery_usage:
                    empty_processed = _RUNTIME.process_response(
                        "",
                        model=recovery_model,
                        session=session.runtime_session,
                        behavior_signals=stream_behavior_signals or None,
                        tool_compatible=tool_compatible,
                    )
                    session.tracker.record_call(
                        model=recovery_model,
                        actual_input=recovery_usage.get("input_tokens", 0),
                        actual_output=recovery_usage.get("output_tokens", 0),
                        cache_read=recovery_usage.get(
                            "cache_read_input_tokens", 0
                        ),
                        cache_write=recovery_usage.get(
                            "cache_creation_input_tokens", 0
                        ),
                        input_saved=input_saved_tokens,
                        output_saved=0,
                        type_breakdown=type_breakdown,
                        behavior_signals=empty_processed.behavior_signals
                        or None,
                        prompt_metrics=prompt_metrics,
                    )
                if request_state is not None:
                    _record_fallback_once(session, request_state)
        if sse_model != "unknown" and sse_usage:
            if not full_text:
                processed = _RUNTIME.process_response(
                    "",
                    model=sse_model,
                    session=session.runtime_session,
                    behavior_signals=stream_behavior_signals or None,
                    tool_compatible=tool_compatible,
                )
                output_saved = 0
                response_signals = processed.behavior_signals
            else:
                output_saved = processed.output_saved_tokens
                response_signals = processed.behavior_signals

            session.tracker.record_call(
                model=sse_model,
                actual_input=sse_usage.get("input_tokens", 0),
                actual_output=sse_usage.get("output_tokens", 0),
                cache_read=sse_usage.get("cache_read_input_tokens", 0),
                cache_write=sse_usage.get("cache_creation_input_tokens", 0),
                input_saved=input_saved_tokens,
                output_saved=output_saved,
                type_breakdown=type_breakdown,
                behavior_signals=response_signals or None,
                prompt_metrics=prompt_metrics,
            )

        content_emitted = False
        for event_str in sse_events:
            if not event_str.strip():
                yield b"\n\n"
                continue

            try:
                data_idx = event_str.find("data: ")
                if data_idx == -1:
                    yield (event_str + "\n\n").encode()
                    continue

                d = json.loads(event_str[data_idx + 6 :])
                etype = d.get("type", "")

                if etype.startswith("content_block_"):
                    if content_blocks:
                        continue
                    yield (event_str + "\n\n").encode()
                    continue

                if (
                    etype in ("message_delta", "message_stop")
                    and not content_emitted
                ):
                    if content_blocks:
                        for i, block in enumerate(content_blocks):
                            block_type = block.get("type")
                            if block_type == "text":
                                start = {
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": {
                                        "type": "text",
                                        "text": "",
                                    },
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode()

                                delta = {
                                    "type": "content_block_delta",
                                    "index": i,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": block["text"],
                                    },
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode()

                                stop = {
                                    "type": "content_block_stop",
                                    "index": i,
                                }
                                yield f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode()
                            else:
                                tool_input = block.get("input", {})
                                start = {
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": block.get("id", ""),
                                        "name": block.get("name", "unknown"),
                                        "input": {},
                                    },
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode()
                                if isinstance(tool_input, dict):
                                    delta = {
                                        "type": "content_block_delta",
                                        "index": i,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": json.dumps(
                                                tool_input
                                            ),
                                        },
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode()
                                stop = {
                                    "type": "content_block_stop",
                                    "index": i,
                                }
                                yield f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode()
                        content_emitted = True

                yield (event_str + "\n\n").encode()

            except (json.JSONDecodeError, KeyError):
                yield (event_str + "\n\n").encode()
    finally:
        response_aclose = getattr(response, "aclose", None)
        if callable(response_aclose):
            await response_aclose()
        client_aclose = getattr(client, "aclose", None)
        if callable(client_aclose):
            await client_aclose()


def create_app_impl(session: BridgeSession | None = None) -> FastAPI:
    """Create the bridge FastAPI application."""
    if session is None:
        session = BridgeSession()

    app = FastAPI(title="tok-bridge")

    async def _send_with_tok_fail_open_retry(
        client: httpx.AsyncClient,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        content: bytes,
        original_content: bytes | None,
        retry_content: bytes | None = None,
        allow_original_retry: bool = True,
        stream: bool = False,
        compressed_request: bool = False,
    ) -> tuple[httpx.Response, bool, dict[str, int]]:
        request_obj = client.build_request(
            method, url, headers=headers, content=content
        )
        response = await client.send(request_obj, stream=stream)
        retried_without_tok = False
        retry_signals: dict[str, int] = {}

        logger.warning(
            "Fail-open check: status=%d, compressed=%s, has_orig=%s, fail_open=%s",
            response.status_code,
            compressed_request,
            original_content is not None,
            session.fail_open,
        )

        if response.status_code == 429:
            retry_signals["rate_limit_retry_started"] = 1
            max_attempts = max(0, session.rate_limit_retry_max_attempts)
            for attempt in range(1, max_attempts + 1):
                session.record_rate_limit_event(time.time())
                retry_after_seconds = _parse_retry_after_seconds(
                    response.headers.get("retry-after")
                )
                computed_backoff_seconds = _compute_rate_limit_backoff_seconds(
                    attempt=attempt,
                    base_ms=session.rate_limit_backoff_base_ms,
                    cap_ms=session.rate_limit_backoff_cap_ms,
                )
                sleep_seconds = max(
                    retry_after_seconds, computed_backoff_seconds
                )
                retry_signals["rate_limit_retry_attempt"] = (
                    retry_signals.get("rate_limit_retry_attempt", 0) + 1
                )
                logger.warning(
                    "rate_limit_retry_attempt: upstream 429 attempt=%d/%d sleep=%.3fs retry_after=%.3fs",
                    attempt,
                    max_attempts,
                    sleep_seconds,
                    retry_after_seconds,
                )
                if stream:
                    await response.aread()
                await response.aclose()
                await asyncio.sleep(sleep_seconds)
                retry_request = client.build_request(
                    method, url, headers=headers, content=content
                )
                response = await client.send(retry_request, stream=stream)
                if response.status_code != 429:
                    retry_signals["rate_limit_retry_succeeded"] = 1
                    break

            if response.status_code == 429:
                session.record_rate_limit_event(time.time())
                retry_signals["rate_limit_retry_exhausted"] = 1
                logger.warning(
                    "rate_limit_retry_exhausted: upstream remained at 429 after %d retry attempts",
                    max_attempts,
                )

        if (
            response.status_code == 400
            and compressed_request
            and session.fail_open
        ):
            fallback_content = retry_content
            if fallback_content is None and allow_original_retry:
                fallback_content = original_content
            if stream:
                await response.aread()
            error_text = response.text
            _log_bridge_body_structure(
                "upstream_400_after_compressed_request",
                content=content,
                headers=headers,
                original_content=original_content,
                compressed_request=compressed_request,
            )
            prepared_summary: dict[str, Any] | str = {}
            prepared_pairing: list[dict[str, Any]] = []
            prepared_failures: list[str] = []
            provider_safe_summary: dict[str, Any] | str = {}
            provider_safe_pairing: list[dict[str, Any]] = []
            provider_safe_failures: list[str] = []
            prepared_mixed_user_tool_result_messages = 0
            prepared_split_boundaries = 0
            provider_safe_mixed_user_tool_result_messages = 0
            provider_safe_split_boundaries = 0
            try:
                prepared_body = json.loads(content)
            except Exception:
                prepared_body = None
            if isinstance(prepared_body, dict):
                prepared_summary = summarize_message_structure(
                    prepared_body.get("messages", [])
                )
                prepared_pairing = summarize_bridge_pairing(
                    prepared_body.get("messages", [])
                )
                prepared_failures = validate_anthropic_bridge_body(prepared_body)
                prepared_mixed_user_tool_result_messages = (
                    _count_user_messages_with_mixed_tool_result_content(
                        prepared_body.get("messages", [])
                    )
                )
                prepared_split_boundaries = (
                    _count_user_tool_result_split_boundaries(
                        prepared_body.get("messages", [])
                    )
                )
            try:
                fallback_body = (
                    json.loads(fallback_content)
                    if fallback_content is not None
                    else None
                )
            except Exception:
                fallback_body = None
            if isinstance(fallback_body, dict):
                provider_safe_summary = summarize_message_structure(
                    fallback_body.get("messages", [])
                )
                provider_safe_pairing = summarize_bridge_pairing(
                    fallback_body.get("messages", [])
                )
                provider_safe_failures = validate_anthropic_bridge_body(
                    fallback_body
                )
                provider_safe_mixed_user_tool_result_messages = (
                    _count_user_messages_with_mixed_tool_result_content(
                        fallback_body.get("messages", [])
                    )
                )
                provider_safe_split_boundaries = (
                    _count_user_tool_result_split_boundaries(
                        fallback_body.get("messages", [])
                    )
                )
            retry_signals["fail_open_retry_prepared_forensics_logged"] = 1
            logger.warning(
                "bridge_pairing_forensics prepared_failures=%s prepared_pairing=%s prepared_summary=%s prepared_mixed_user_tool_result_messages=%s prepared_split_boundaries=%s provider_safe_failures=%s provider_safe_pairing=%s provider_safe_summary=%s provider_safe_mixed_user_tool_result_messages=%s provider_safe_split_boundaries=%s",
                prepared_failures,
                prepared_pairing,
                prepared_summary,
                prepared_mixed_user_tool_result_messages,
                prepared_split_boundaries,
                provider_safe_failures,
                provider_safe_pairing,
                provider_safe_summary,
                provider_safe_mixed_user_tool_result_messages,
                provider_safe_split_boundaries,
            )
            if (
                "`tool_use` ids were found without `tool_result` blocks immediately after"
                in error_text
                and not prepared_failures
            ):
                retry_signals[
                    "fail_open_retry_upstream_pairing_disagreement"
                ] = 1
                if prepared_mixed_user_tool_result_messages > 0:
                    retry_signals[
                        "fail_open_retry_upstream_pairing_disagreement_mixed_user_message_present"
                    ] = 1
                    logger.warning(
                        "fail_open_retry_upstream_pairing_disagreement_mixed_user_message_present: prepared payload still contained mixed user tool_result+non-tool blocks"
                    )
                elif prepared_split_boundaries > 0:
                    retry_signals[
                        "fail_open_retry_upstream_pairing_disagreement_after_user_message_split"
                    ] = 1
                    logger.warning(
                        "fail_open_retry_upstream_pairing_disagreement_after_user_message_split: prepared payload had user tool_result/text split boundaries"
                    )
                else:
                    retry_signals[
                        "fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message"
                    ] = 1
                    logger.warning(
                        "fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message: no mixed user message or split boundary detected in prepared payload"
                    )
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement: upstream reported pairing failure while local strict validation passed (prepared_mixed_user_tool_result_messages=%s prepared_split_boundaries=%s)",
                    prepared_mixed_user_tool_result_messages,
                    prepared_split_boundaries,
                )
            if fallback_content is not None and _payloads_materially_differ(
                content, fallback_content
            ):
                if provider_safe_failures:
                    logger.warning(
                        "fail_open_retry_provider_safe_invalid: refusing retry because provider-safe payload failed local strict validation: %s",
                        provider_safe_failures,
                    )
                    retry_signals["fail_open_retry_provider_safe_invalid"] = 1
                    return response, retried_without_tok, retry_signals
                retry_signals["fail_open_retry_provider_safe_validated"] = 1
                logger.info(
                    "fail_open_retry_provider_safe_validated: provider-safe fallback passed local strict validation"
                )
                retry_kind = (
                    "provider-safe"
                    if retry_content is not None
                    else "original"
                )
                logger.warning(
                    "Upstream 400 after Tok request preparation: %s; retrying with %s payload",
                    error_text[:500],
                    retry_kind,
                )
                await response.aclose()
                request_obj = client.build_request(
                    method, url, headers=headers, content=fallback_content
                )
                response = await client.send(request_obj, stream=stream)
                retried_without_tok = True
                if retry_content is not None:
                    retry_signals["fail_open_retry_provider_safe"] = 1
                if not allow_original_retry:
                    retry_signals["fail_open_raw_retry_blocked"] = 1
            elif not allow_original_retry and original_content is not None:
                logger.warning(
                    "fail_open_raw_retry_blocked: refusing to resend raw original payload after tool-history repair"
                )
                retry_signals["fail_open_raw_retry_blocked"] = 1
        return response, retried_without_tok, retry_signals

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root() -> dict[str, str]:
        return {"status": "ok", "bridge": "tok"}

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health() -> dict[str, Any]:
        session_summary = session.tracker.session_summary() or {}
        signals = session.tracker.behavior_signals()
        return {
            "status": "ok",
            "bridge": "tok",
            "port": session.port,
            "mode": "tool-compatible"
            if session.tool_compatible_default
            else "baseline",
            "baseline_only": session.runtime_session._baseline_only,
            "fallback_count": int(
                session_summary.get(
                    "fallback_count",
                    session.tracker.behavior_signals().get(
                        "tok_fallback_activated", 0
                    ),
                )
            ),
            "actual_tokens": int(session_summary.get("actual_tokens", 0)),
            "baseline_tokens": int(session_summary.get("baseline_tokens", 0)),
            "session_tokens_saved": int(
                session_summary.get("tokens_saved", 0)
            ),
            "baseline_prompt_tokens": int(
                session_summary.get("baseline_prompt_tokens", 0)
            ),
            "prepared_prompt_tokens": int(
                session_summary.get("prepared_prompt_tokens", 0)
            ),
            "saved_prompt_tokens": int(
                session_summary.get("saved_prompt_tokens", 0)
            ),
            "session_savings_pct": float(
                session_summary.get("savings_pct", 0.0)
            ),
            "actual_cost_usd": float(
                session_summary.get("actual_cost_usd", 0.0)
            ),
            "baseline_cost_usd": float(
                session_summary.get("baseline_cost_usd", 0.0)
            ),
            "cost_saved_usd": float(
                session_summary.get("cost_saved_usd", 0.0)
            ),
            "semantic_drift_count": int(
                session_summary.get(
                    "semantic_drift_count",
                    signals.get("semantic_drift_detected", 0),
                )
            ),
            "fail_open_count": int(
                session_summary.get(
                    "fail_open_count",
                    signals.get("fail_open_compat_response", 0),
                )
            ),
            "non_tok_count": int(
                session_summary.get(
                    "non_tok_count", signals.get("non_tok_response", 0)
                )
            ),
            "answer_anchor_miss_count": int(
                session_summary.get("answer_anchor_miss_count", 0)
            ),
            "repeat_search_count": int(signals.get("repeat_search", 0)),
            "repeat_file_read_count": int(signals.get("repeat_file_read", 0)),
            "shell_file_read_normalized_count": int(
                signals.get("shell_file_read_normalized", 0)
            ),
            "shell_file_snapshot_captured_count": int(
                signals.get("shell_file_snapshot_captured", 0)
            ),
            "repeat_target_hot_count": int(
                signals.get("repeat_target_hot", 0)
            ),
            "repeat_target_stuck_count": int(
                signals.get("repeat_target_stuck", 0)
            ),
            "hot_recent_hint_count": int(
                signals.get("hot_recent_hint_injected", 0)
            ),
            "hot_hint_tokens_added": int(
                session_summary.get(
                    "hot_hint_tokens_added",
                    signals.get("hot_hint_tokens_added", 0),
                )
            ),
            "reacquisition_tokens_avoided_estimate": int(
                session_summary.get(
                    "reacquisition_tokens_avoided_estimate",
                    signals.get("reacquisition_tokens_avoided_estimate", 0),
                )
            ),
            "state_resend_full_count": int(
                signals.get("state_resend_full_turn", 0)
            ),
            "state_resend_delta_count": int(
                signals.get("state_resend_delta_turn", 0)
            ),
            "state_resend_suppressed_count": int(
                signals.get("state_resend_suppressed_turn", 0)
            ),
            "stream_recovery_attempt_count": int(
                session_summary.get("stream_recovery_attempt_count", 0)
            ),
            "stream_recovery_success_text_count": int(
                session_summary.get("stream_recovery_success_text_count", 0)
            ),
            "stream_recovery_success_tool_use_count": int(
                session_summary.get(
                    "stream_recovery_success_tool_use_count", 0
                )
            ),
            "stream_recovery_fallback_count": int(
                session_summary.get("stream_recovery_fallback_count", 0)
            ),
            "tool_history_repaired_count": int(
                session_summary.get("tool_history_repaired_count", 0)
            ),
            "tool_history_pairing_repaired_count": int(
                session_summary.get(
                    "tool_history_pairing_repaired_count", 0
                )
            ),
            "tool_history_quarantined_count": int(
                session_summary.get("tool_history_quarantined_count", 0)
            ),
            "tool_history_blocked_count": int(
                session_summary.get("tool_history_blocked_count", 0)
            ),
            "invalid_tool_history_session_reset_count": int(
                session_summary.get(
                    "invalid_tool_history_session_reset_count", 0
                )
            ),
            "session_quality": str(
                session_summary.get("session_quality", "clean")
            ),
            "last_degradation_reason": str(
                session_summary.get("last_degradation_reason", "")
            ),
        }

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def bridge(request: Request, path: str) -> Response:
        body_bytes = await request.body()
        original_body_bytes = body_bytes
        request_state = {"fallback_recorded": False}

        skip = {
            "host",
            "content-length",
            "transfer-encoding",
            "connection",
            "x-tok-tool-compatible",
        }
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in skip
        }
        headers["host"] = "api.anthropic.com"

        has_x_api_key = any(k.lower() == "x-api-key" for k in headers)
        has_auth_bearer = any(
            k.lower() == "authorization" and v.lower().startswith("bearer ")
            for k, v in headers.items()
        )

        if path.startswith("v1/") and not (has_x_api_key or has_auth_bearer):
            logger.warning("Blocking request to %s — missing API key", path)
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "authentication_error",
                            "message": "x-api-key or Authorization (Bearer) header is required",
                        }
                    }
                ),
                status_code=401,
                media_type="application/json",
            )

        compressed = False
        saved_toks = 0
        tool_breakdown: dict[str, int] = {}
        behavior_signals: dict[str, int] = {}
        prompt_metrics = {
            "baseline_prompt_tokens": 0,
            "prepared_prompt_tokens": 0,
            "saved_prompt_tokens": 0,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
        }
        request_tool_compatible = False
        provider_safe_original_body_bytes = original_body_bytes
        raw_retry_forbidden = False

        if (
            path in {"v1/messages", "v1/messages/count_tokens"}
            and request.method == "POST"
            and body_bytes
        ):
            try:
                body = json.loads(body_bytes)
                original_body = (
                    copy.deepcopy(body) if isinstance(body, dict) else body
                )
                if not isinstance(body, dict) or not isinstance(
                    original_body, dict
                ):
                    raise ValueError("request body must be a JSON object")

                provider_safe_original_body, behavior_signals, source_retry_forbidden, preflight_response = _run_bridge_preflight(
                    session,
                    body=copy.deepcopy(body),
                    original_body=original_body,
                    headers=headers,
                    behavior_signals=behavior_signals,
                    compressed=False,
                    request_state=request_state,
                    path=path,
                    emit_ready_log=(path == "v1/messages/count_tokens"),
                    emit_repair_logs=(path == "v1/messages/count_tokens"),
                    reset_recovery_state=(path == "v1/messages/count_tokens"),
                )
                if preflight_response is not None:
                    return preflight_response
                provider_safe_original_body_bytes = json.dumps(
                    provider_safe_original_body
                ).encode()
                raw_retry_forbidden = source_retry_forbidden
                source_behavior_signals = dict(behavior_signals)

                request_model = str(provider_safe_original_body.get("model", ""))
                messages = provider_safe_original_body.get("messages", [])

                if path == "v1/messages":
                    tok_tool_header = request.headers.get(
                        "x-tok-tool-compatible", ""
                    )
                    if tok_tool_header.lower() in {
                        "0",
                        "false",
                        "off",
                        "no",
                    }:
                        request_tool_compatible = False
                    elif session.runtime_session._baseline_only:
                        request_tool_compatible = False
                        behavior_signals["baseline_only_session"] = 1
                        behavior_signals["tok_fallback_activated"] = 1
                        logger.warning(
                            "tok_fallback_activated: session is in baseline-only mode, serving without compression"
                        )
                    else:
                        request_tool_compatible = (
                            session.tool_compatible_default
                        )

                    logger.info(
                        "Request mode: model=%s, tool_compatible=%s (tools present: %s, header=%s)",
                        request_model,
                        request_tool_compatible,
                        bool(provider_safe_original_body.get("tools")),
                        tok_tool_header or "<unset>",
                    )

                    prepared = _RUNTIME.prepare_request(
                        RuntimeRequest(
                            model=request_model,
                            messages=messages,
                            system=provider_safe_original_body.get(
                                "system", ""
                            ),
                            adapter_kind="claude-bridge",
                            tool_compatible=request_tool_compatible,
                        ),
                        session.runtime_session,
                        result_cache=session.result_cache,
                    )
                    compressed = prepared.compressed
                    saved_toks = prepared.input_saved_tokens
                    tool_breakdown = prepared.type_breakdown
                    behavior_signals = dict(prepared.behavior_signals)
                    _merge_signal_counts(
                        behavior_signals, source_behavior_signals
                    )
                    prompt_metrics = {
                        "baseline_prompt_tokens": prepared.baseline_prompt_tokens,
                        "prepared_prompt_tokens": prepared.prepared_prompt_tokens,
                        "saved_prompt_tokens": prepared.saved_prompt_tokens,
                        "hot_hint_tokens_added": prepared.hot_hint_tokens_added,
                        "reacquisition_tokens_avoided_estimate": prepared.reacquisition_tokens_avoided_estimate,
                    }
                    body = copy.deepcopy(provider_safe_original_body)
                    body["messages"] = prepared.body.get("messages", [])
                    body["system"] = prepared.body.get(
                        "system", body.get("system", "")
                    )

                    session.capture_request(
                        {
                            "event": "request",
                            "messages": messages,
                            "system": body.get("system", ""),
                            "model": request_model,
                            "tool_compatible": request_tool_compatible,
                        }
                    )

                    (
                        body,
                        behavior_signals,
                        prepared_retry_forbidden,
                        preflight_response,
                    ) = _run_bridge_preflight(
                        session,
                        body=body,
                        original_body=provider_safe_original_body,
                        headers=headers,
                        behavior_signals=behavior_signals,
                        compressed=compressed,
                        request_state=request_state,
                        path=path,
                    )
                    if preflight_response is not None:
                        return preflight_response
                    raw_retry_forbidden = (
                        raw_retry_forbidden or prepared_retry_forbidden
                    )
                    if behavior_signals.get(
                        "tok_bridge_pairing_degraded_to_provider_safe", 0
                    ):
                        compressed = False
                        saved_toks = 0
                        tool_breakdown = {}
                        prompt_metrics = {
                            "baseline_prompt_tokens": 0,
                            "prepared_prompt_tokens": 0,
                            "saved_prompt_tokens": 0,
                            "hot_hint_tokens_added": 0,
                            "reacquisition_tokens_avoided_estimate": 0,
                        }
                    body_bytes = json.dumps(body).encode()

                    if tool_breakdown:
                        logger.info(
                            "Tool results: ~%d tokens saved %s",
                            sum(tool_breakdown.values()) // 4,
                            {k: v // 4 for k, v in tool_breakdown.items()},
                        )
                    if compressed:
                        logger.info(
                            "Prepared request via runtime | ~%d tokens saved",
                            saved_toks,
                        )
                        session.runtime_session.reset_fallback_count()
                else:
                    body = provider_safe_original_body
                    body_bytes = provider_safe_original_body_bytes
                    session.capture_request(
                        {
                            "event": "count_tokens_request",
                            "messages": messages,
                            "system": body.get("system", ""),
                            "model": request_model,
                            "tool_compatible": False,
                        }
                    )

            # Handle specific exception types for better error classification
            except (json.JSONDecodeError, ValueError) as exc:
                if session.fail_open:
                    logger.warning(
                        "tok_fallback_activated: JSON decode error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["json_decode_error"] = 1
                    _record_fallback_once(session, request_state)
                else:
                    raise
            except (KeyError, AttributeError) as exc:
                if session.fail_open:
                    logger.warning(
                        "tok_fallback_activated: data structure error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["data_structure_error"] = 1
                    _record_fallback_once(session, request_state)
                else:
                    raise
            except (MemoryError, OverflowError) as exc:
                # Critical system errors - always fallback
                logger.error(
                    "tok_fallback_activated: critical system error, serving without compression: %s",
                    exc,
                )
                behavior_signals["processing_error"] = 1
                behavior_signals["tok_fallback_activated"] = 1
                behavior_signals["critical_system_error"] = 1
                _record_fallback_once(session, request_state)
            except Exception as exc:
                # Catch-all for truly unexpected errors
                if session.fail_open:
                    logger.error(
                        "tok_fallback_activated: unexpected error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["unexpected_error"] = 1
                    _record_fallback_once(session, request_state)
                else:
                    logger.error(
                        "Unexpected error in request processing: %s", exc
                    )
                    raise

        target_url = f"{ANTHROPIC_API_BASE}/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        if session.is_rate_limited_locally():
            retry_after_seconds = session.local_rate_limit_retry_after_seconds()
            behavior_signals["rate_limit_local_throttle_active"] = 1
            logger.warning(
                "rate_limit_local_throttle_active: blocking upstream request during local cooldown retry_after=%ss",
                retry_after_seconds,
            )
            return _local_rate_limit_response(retry_after_seconds)

        is_streaming = False
        try:
            body_dict = json.loads(body_bytes)
            is_streaming = bool(body_dict.get("stream", False))
        except (json.JSONDecodeError, ValueError):
            # Invalid JSON - not streaming
            logger.debug("Failed to parse JSON for streaming detection")
        except Exception as exc:
            # Other unexpected errors during JSON parsing
            logger.debug("Unexpected error during JSON parsing: %s", exc)

        if is_streaming:
            # Use context manager for proper resource cleanup
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    (
                        response,
                        retried_without_tok,
                        retry_signals,
                    ) = await _send_with_tok_fail_open_retry(
                        client,
                        method=request.method,
                        url=target_url,
                        headers=headers,
                        content=body_bytes,
                        original_content=original_body_bytes,
                        retry_content=provider_safe_original_body_bytes,
                        allow_original_retry=not raw_retry_forbidden,
                        stream=True,
                        compressed_request=compressed,
                    )
                    _merge_signal_counts(behavior_signals, retry_signals)
                    if session.is_rate_limited_locally():
                        behavior_signals["rate_limit_local_throttle_opened"] = 1
                    if retried_without_tok:
                        compressed = False
                        saved_toks = 0
                        tool_breakdown = {}
                        prompt_metrics = {
                            "baseline_prompt_tokens": 0,
                            "prepared_prompt_tokens": 0,
                            "saved_prompt_tokens": 0,
                            "hot_hint_tokens_added": 0,
                            "reacquisition_tokens_avoided_estimate": 0,
                        }
                        behavior_signals = {
                            "tok_fail_open_retry": 1,
                            "tok_fallback_activated": 1,
                        }
                        _merge_signal_counts(
                            behavior_signals, retry_signals
                        )
                        logger.warning(
                            "tok_fallback_activated: upstream 400 retry, serving without compression"
                        )
                        _record_fallback_once(session, request_state)
                    resp_headers = _safe_headers(response.headers)
                    if response.status_code >= 400:
                        error_content = await response.aread()
                        return Response(
                            content=error_content,
                            status_code=response.status_code,
                            headers=resp_headers,
                            media_type=response.headers.get(
                                "content-type", "application/json"
                            ),
                        )

                    if path == "v1/messages":
                        return StreamingResponse(
                            buffer_strip_restream_impl(
                                session,
                                client,
                                response,
                                input_saved_tokens=saved_toks
                                if compressed
                                else 0,
                                type_breakdown=tool_breakdown
                                if compressed
                                else None,
                                behavior_signals=behavior_signals or None,
                                prompt_metrics=prompt_metrics
                                if compressed
                                else None,
                                tool_compatible=request_tool_compatible,
                                request_method=request.method,
                                request_url=target_url,
                                request_headers=headers,
                                request_content=(
                                    provider_safe_original_body_bytes
                                    if retried_without_tok
                                    else body_bytes
                                ),
                                request_state=request_state,
                            ),
                            status_code=response.status_code,
                            headers=resp_headers,
                            media_type=response.headers.get(
                                "content-type", "text/event-stream"
                            ),
                        )
                except Exception as e:
                    logger.error(
                        "Streaming error in Tok bridge: %s",
                        str(e),
                        exc_info=True,
                    )
                    if session.fail_open:
                        logger.warning(
                            "Streaming error - fail-open: retrying without Tok"
                        )
                        compressed = False
                        saved_toks = 0
                        tool_breakdown = {}
                        behavior_signals = {
                            "streaming_error_retry": 1,
                        }
                        async with httpx.AsyncClient(
                            timeout=300.0
                        ) as retry_client:
                            (
                                response,
                                retried,
                                retry_signals,
                            ) = await _send_with_tok_fail_open_retry(
                                retry_client,
                                method=request.method,
                                url=target_url,
                                headers=headers,
                                content=provider_safe_original_body_bytes,
                                original_content=original_body_bytes,
                                retry_content=provider_safe_original_body_bytes,
                                allow_original_retry=not raw_retry_forbidden,
                                stream=True,
                                compressed_request=False,
                            )
                            _merge_signal_counts(
                                behavior_signals, retry_signals
                            )
                            if path == "v1/messages":
                                return StreamingResponse(
                                    buffer_strip_restream_impl(
                                        session,
                                        retry_client,
                                        response,
                                        input_saved_tokens=0,
                                        type_breakdown=None,
                                        behavior_signals=behavior_signals,
                                        prompt_metrics=None,
                                        tool_compatible=request_tool_compatible,
                                        request_method=request.method,
                                        request_url=target_url,
                                        request_headers=headers,
                                        request_content=provider_safe_original_body_bytes,
                                        request_state=request_state,
                                    ),
                                    status_code=response.status_code,
                                    headers=_safe_headers(response.headers),
                                    media_type=response.headers.get(
                                        "content-type", "text/event-stream"
                                    ),
                                )
                    return Response(
                        content=f"Streaming error: {str(e)}",
                        status_code=502,
                        media_type="text/plain",
                    )

        async with httpx.AsyncClient(timeout=300.0) as client:
            (
                response,
                retried_without_tok,
                retry_signals,
            ) = await _send_with_tok_fail_open_retry(
                client,
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes,
                original_content=original_body_bytes,
                retry_content=provider_safe_original_body_bytes,
                allow_original_retry=not raw_retry_forbidden,
                compressed_request=compressed,
            )
            _merge_signal_counts(behavior_signals, retry_signals)
            if session.is_rate_limited_locally():
                behavior_signals["rate_limit_local_throttle_opened"] = 1
            if retried_without_tok:
                compressed = False
                saved_toks = 0
                tool_breakdown = {}
                prompt_metrics = {
                    "baseline_prompt_tokens": 0,
                    "prepared_prompt_tokens": 0,
                    "saved_prompt_tokens": 0,
                    "hot_hint_tokens_added": 0,
                    "reacquisition_tokens_avoided_estimate": 0,
                }
                behavior_signals = {
                    "tok_fail_open_retry": 1,
                    "tok_fallback_activated": 1,
                }
                _merge_signal_counts(behavior_signals, retry_signals)
                logger.warning(
                    "tok_fallback_activated: upstream 400 retry, serving without compression"
                )
                _record_fallback_once(session, request_state)

            if response.status_code >= 400:
                logger.warning(
                    "Upstream %d: %s",
                    response.status_code,
                    response.text[:300],
                )

            content = response.content
            if path == "v1/messages" and response.status_code == 200:
                try:
                    resp_json = json.loads(content)
                    total_output_saved = 0
                    full_response_text = ""
                    passthrough_blocks = [
                        block
                        for block in resp_json.get("content", [])
                        if isinstance(block, dict) and block.get("type") != "text"
                    ]
                    passthrough_blocks, passthrough_signals = (
                        normalize_tool_use_blocks(
                            passthrough_blocks, seed_prefix="toolu_upstream"
                        )
                    )
                    _merge_signal_counts(behavior_signals, passthrough_signals)

                    logger.info(
                        "Raw response content: %s",
                        resp_json.get("content", [])[:3] if resp_json.get("content") else [],
                    )

                    for block in resp_json.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content = block.get("text")
                            if isinstance(text_content, str):
                                full_response_text += text_content

                    response_signals: dict[str, int] = {}
                    if full_response_text:
                        processed = _RUNTIME.process_response(
                            full_response_text,
                            model=str(resp_json.get("model", "")),
                            session=session.runtime_session,
                            behavior_signals=behavior_signals or None,
                            tool_compatible=request_tool_compatible,
                        )
                        response_signals = processed.behavior_signals
                        new_content = (
                            processed.content_blocks + passthrough_blocks
                        )
                        resp_json["content"] = new_content
                        total_output_saved = processed.output_saved_tokens

                        session_signals = session.consume_behavior_signals()
                        if session_signals:
                            response_signals = response_signals or {}
                            for k, v in session_signals.items():
                                response_signals[k] = (
                                    response_signals.get(k, 0) + v
                                )

                        logger.info(
                            "Response: %d blocks, ~%d saved",
                            len(new_content),
                            total_output_saved,
                        )

                    usage = resp_json.get("usage", {})
                    if resp_json.get("model") and usage:
                        session.tracker.record_call(
                            model=resp_json["model"],
                            actual_input=usage.get("input_tokens", 0),
                            actual_output=usage.get("output_tokens", 0),
                            cache_read=usage.get("cache_read_input_tokens", 0),
                            cache_write=usage.get(
                                "cache_creation_input_tokens", 0
                            ),
                            input_saved=saved_toks if compressed else 0,
                            output_saved=total_output_saved,
                            type_breakdown=tool_breakdown
                            if compressed
                            else None,
                            behavior_signals=response_signals or None,
                            prompt_metrics=prompt_metrics
                            if compressed
                            else None,
                        )
                        session.capture_event(
                            {
                                "event": "response",
                                "model": resp_json["model"],
                                "baseline_only": session.runtime_session._baseline_only,
                                "fallback_count": int(
                                    session.tracker.behavior_signals().get(
                                        "tok_fallback_activated", 0
                                    )
                                ),
                                "behavior_signals": response_signals or {},
                                "session_quality": str(
                                    (
                                        session.tracker.session_summary() or {}
                                    ).get("session_quality", "clean")
                                ),
                                "session_tokens_saved": int(
                                    (
                                        session.tracker.session_summary() or {}
                                    ).get("tokens_saved", 0)
                                ),
                                "session_savings_pct": float(
                                    (
                                        session.tracker.session_summary() or {}
                                    ).get("savings_pct", 0.0)
                                ),
                                "last_degradation_reason": str(
                                    (
                                        session.tracker.session_summary() or {}
                                    ).get("last_degradation_reason", "")
                                ),
                            }
                        )
                    content = json.dumps(resp_json).encode()
                except Exception as exc:
                    if session.fail_open:
                        logger.warning(
                            "Non-streaming processing error (fail-open): %s",
                            exc,
                        )
                        try:
                            _model = resp_json.get("model", "")
                            _usage = resp_json.get("usage", {})
                            session_signals = (
                                session.consume_behavior_signals()
                            )
                            error_signals = {"processing_error": 1}
                            if session_signals:
                                for k, v in session_signals.items():
                                    error_signals[k] = (
                                        error_signals.get(k, 0) + v
                                    )

                            if _model and _usage:
                                session.tracker.record_call(
                                    model=_model,
                                    actual_input=_usage.get("input_tokens", 0),
                                    actual_output=_usage.get(
                                        "output_tokens", 0
                                    ),
                                    cache_read=_usage.get(
                                        "cache_read_input_tokens", 0
                                    ),
                                    cache_write=_usage.get(
                                        "cache_creation_input_tokens", 0
                                    ),
                                    input_saved=saved_toks
                                    if compressed
                                    else 0,
                                    output_saved=0,
                                    type_breakdown=tool_breakdown
                                    if compressed
                                    else None,
                                    behavior_signals=error_signals,
                                    prompt_metrics=prompt_metrics
                                    if compressed
                                    else None,
                                )
                        except Exception as _exc:
                            logger.debug(
                                "Failed to record usage in fail-open path: %s",
                                _exc,
                            )
                    else:
                        raise

            return Response(
                content=content,
                status_code=response.status_code,
                headers=_safe_headers(response.headers),
                media_type=response.headers.get(
                    "content-type", "application/json"
                ),
            )

    return app
