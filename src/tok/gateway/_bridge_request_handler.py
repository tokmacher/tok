"""Fail-open request helpers for the Tok gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
import time
from typing import TYPE_CHECKING, Any

import httpx

from tok.runtime.pipeline.request_validation import (
    has_blocking_outgoing_failures,
    has_provider_sensitive_failures,
    has_recoverable_immediate_pairing_failures,
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_bridge_body,
    validate_anthropic_outgoing_bridge_body,
)

from . import BridgeSession, _log_bridge_body_structure, logger
from ._bridge_comparison import _payloads_materially_differ
from ._bridge_preflight import (
    _count_user_messages_with_mixed_tool_result_content,
    _count_user_tool_result_split_boundaries,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx

__all__ = ["send_with_tok_fail_open_retry"]


def _normalize_provider_safe_retry_payload(
    body: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Normalize provider-safe retry payload to remove thinking blocks between tool_use blocks.

    This handles the case where an assistant message has interleaved thinking/redacted_thinking
    blocks between tool_use blocks, which can cause upstream pairing failures.

    Returns (normalized_body, changed) where changed is True if any thinking blocks were removed.
    """
    if not isinstance(body, dict):
        return body, False

    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, False

    changed = False
    normalized_messages: list[dict[str, Any]] = []

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            normalized_messages.append(msg)
            continue

        role = str(msg.get("role", "")).strip()
        content = msg.get("content")

        if role != "assistant" or not isinstance(content, list):
            normalized_messages.append(msg)
            continue

        has_tool_use = any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)

        if not has_tool_use:
            normalized_messages.append(msg)
            continue

        is_current_turn = all(m.get("role") != "assistant" for m in messages[msg_idx + 1 :])

        if is_current_turn:
            message_changed = False
            filtered_content: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"}:
                    message_changed = True
                    changed = True
                    continue
                filtered_content.append(block)
        else:
            first_tool_index = next(
                (
                    block_index
                    for block_index, block in enumerate(content)
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ),
                None,
            )
            message_changed = False
            filtered_content = []
            for block_index, block in enumerate(content):
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"thinking", "redacted_thinking"}
                    and first_tool_index is not None
                    and block_index > first_tool_index
                ):
                    message_changed = True
                    changed = True
                    continue
                filtered_content.append(block)

        if message_changed:
            new_msg = dict(msg.items())
            new_msg["content"] = filtered_content
            normalized_messages.append(new_msg)
        else:
            normalized_messages.append(msg)

    if not changed:
        return body, False

    new_body = dict(body.items())
    new_body["messages"] = normalized_messages
    return new_body, True


def _decode_bridge_body(raw_content: bytes | None) -> dict[str, Any] | None:
    if raw_content is None:
        return None
    try:
        decoded = json.loads(raw_content)
    except Exception:
        logger.debug("Failed to decode bridge body as JSON", exc_info=True)
        return None
    return decoded if isinstance(decoded, dict) else None


def _validate_outgoing_bridge_body_from_bytes(
    raw_content: bytes | None,
) -> list[str]:
    body = _decode_bridge_body(raw_content)
    if not isinstance(body, dict):
        return ["body_not_dict"]
    return validate_anthropic_outgoing_bridge_body(body)


def _rate_limit_retry_delay_seconds(session: BridgeSession, *, attempt_index: int, retry_after: str | None) -> float:
    retry_after_delay = 0.0
    if retry_after:
        try:
            parsed_retry_after = float(retry_after)
            if parsed_retry_after > 0.0:
                retry_after_delay = parsed_retry_after
        except ValueError:
            retry_after_delay = 0.0

    backoff_seconds: float = min(
        session.rate_limit_backoff_cap_ms / 1000.0,
        (session.rate_limit_backoff_base_ms / 1000.0) * (2**attempt_index),
    )
    if retry_after_delay > backoff_seconds:
        return retry_after_delay
    return backoff_seconds


async def send_with_tok_fail_open_retry(
    session: BridgeSession,
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
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[httpx.Response, bool, dict[str, int]]:
    if sleep_fn is None:
        sleep_fn = asyncio.sleep
    retried_without_tok = False
    retry_signals: dict[str, int] = {}

    if session._rate_limit_throttle_until > time.time():
        remaining = session._rate_limit_throttle_until - time.time()
        logger.info(
            "rate_limit_cooldown_skip: deferring request, cooldown active (%.2fs remaining)",
            remaining,
        )
        retry_signals["rate_limit_cooldown_skipped"] = 1
        return (
            httpx.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "Tok cooldown active"}},
                headers={"Retry-After": str(max(1, int(math.ceil(remaining))))},
            ),
            False,
            retry_signals,
        )

    request_obj = client.build_request(method, url, headers=headers, content=content)
    response = await client.send(request_obj, stream=stream)
    retry_attempts = 0

    while response.status_code == 429 and retry_attempts < max(0, session.rate_limit_retry_max_attempts):
        async with session._rate_limit_lock:
            if session._rate_limit_retry_owner:
                logger.info("rate_limit_retry_yield: another request already retrying, returning 429 to client")
                retry_signals["rate_limit_retry_yielded"] = 1
                break

            session._rate_limit_retry_owner = True

        try:
            retry_after = response.headers.get("retry-after")
            delay = _rate_limit_retry_delay_seconds(session, attempt_index=retry_attempts, retry_after=retry_after)
            logger.warning(
                "rate_limit_retry_attempt: attempt=%d delay=%.2f retry_after=%s",
                retry_attempts + 1,
                delay,
                retry_after or "unknown",
            )
            retry_signals["rate_limit_retry_attempt"] = retry_signals.get("rate_limit_retry_attempt", 0) + 1
            await response.aclose()

            async with session._rate_limit_lock:
                session._rate_limit_throttle_until = time.time() + delay

            await sleep_fn(delay)
            jitter = random.uniform(0.05, 0.2)
            await sleep_fn(jitter)

            request_obj = client.build_request(method, url, headers=headers, content=content)
            response = await client.send(request_obj, stream=stream)
            retry_attempts += 1
        finally:
            async with session._rate_limit_lock:
                if response.status_code == 200:
                    session._rate_limit_throttle_until = 0.0
                session._rate_limit_retry_owner = False

    if response.status_code == 429:
        retry_after = response.headers.get("retry-after", "unknown")
        logger.warning(
            "rate_limit_retry_exhausted: retries exhausted after %d attempt(s) (retry_after=%s)",
            retry_attempts,
            retry_after,
        )
        retry_signals["rate_limit_retry_exhausted"] = 1

    # This is an invariant/health assertion, not an error event. Keep it visible at
    # INFO so it doesn't read like a triggered fallback during normal operation.
    logger.info(
        "Fail-open check: status=%d, compressed=%s, has_orig=%s, fail_open=%s",
        response.status_code,
        compressed_request,
        original_content is not None,
        session.fail_open,
    )

    if response.status_code == 429:
        # Pass through to client - let Claude Code handle quota exhaustion
        retry_after = response.headers.get("retry-after", "unknown")
        logger.warning(
            "rate_limit_429: upstream returned 429, passing through to client (retry_after=%s)",
            retry_after,
        )
        retry_signals["rate_limit_429_passed_through"] = 1

    if response.status_code == 400 and compressed_request and session.fail_open:
        from tok.runtime.smoothness.models import TokMode

        current_mode = None
        with contextlib.suppress(Exception):
            current_mode = session.runtime_session.current_tok_mode

        skip_provider_safe_recanonicalization = current_mode in (
            TokMode.SMOOTH_MODE,
            TokMode.LOSSLESS_TASK_MODE,
        )
        if skip_provider_safe_recanonicalization:
            if (
                allow_original_retry
                and original_content is not None
                and _payloads_materially_differ(content, original_content)
            ):
                logger.warning(
                    "SMOOTH_MODE active: skipping provider-safe recanonicalization, retrying with original payload"
                )
                await response.aclose()
                request_obj = client.build_request(method, url, headers=headers, content=original_content)
                response = await client.send(request_obj, stream=stream)
                retried_without_tok = True
                retry_signals["fail_open_smooth_mode_original_retry"] = 1
                return response, retried_without_tok, retry_signals

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
        prepared_body = _decode_bridge_body(content)
        if isinstance(prepared_body, dict):
            prepared_summary = summarize_message_structure(prepared_body.get("messages", []))
            prepared_pairing = summarize_bridge_pairing(prepared_body.get("messages", []))
            prepared_failures = validate_anthropic_bridge_body(prepared_body)
            prepared_mixed_user_tool_result_messages = _count_user_messages_with_mixed_tool_result_content(
                prepared_body.get("messages", [])
            )
            prepared_split_boundaries = _count_user_tool_result_split_boundaries(prepared_body.get("messages", []))
        fallback_body = _decode_bridge_body(fallback_content)
        # Normalize provider-safe retry payload to remove thinking blocks between tool_use blocks
        if isinstance(fallback_body, dict):
            (
                normalized_fallback_body,
                normalized_changed,
            ) = _normalize_provider_safe_retry_payload(fallback_body)
            if normalized_changed:
                fallback_body = normalized_fallback_body
                fallback_content = json.dumps(fallback_body).encode()
                retry_signals["provider_safe_removed_assistant_thinking_between_tool_use"] = 1
                logger.warning(
                    "provider_safe_removed_assistant_thinking_between_tool_use: removed thinking/redacted_thinking blocks from provider-safe retry payload"
                )
        if isinstance(fallback_body, dict):
            provider_safe_summary = summarize_message_structure(fallback_body.get("messages", []))
            provider_safe_pairing = summarize_bridge_pairing(fallback_body.get("messages", []))
            provider_safe_failures = validate_anthropic_bridge_body(fallback_body)
            provider_safe_mixed_user_tool_result_messages = _count_user_messages_with_mixed_tool_result_content(
                fallback_body.get("messages", [])
            )
            provider_safe_split_boundaries = _count_user_tool_result_split_boundaries(fallback_body.get("messages", []))
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
        retry_candidate_content = fallback_content
        retry_candidate_kind = "provider-safe" if retry_content is not None else "original"
        retry_candidate_valid = True
        if (
            "`tool_use` ids were found without `tool_result` blocks immediately after" in error_text
            and not prepared_failures
        ):
            retry_signals["fail_open_retry_upstream_pairing_disagreement"] = 1
            if prepared_mixed_user_tool_result_messages > 0:
                retry_signals["fail_open_retry_upstream_pairing_disagreement_mixed_user_message_present"] = 1
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement_mixed_user_message_present: prepared payload still contained mixed user tool_result+non-tool blocks"
                )
            elif prepared_split_boundaries > 0:
                retry_signals["fail_open_retry_upstream_pairing_disagreement_after_user_message_split"] = 1
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement_after_user_message_split: prepared payload had user tool_result/text split boundaries"
                )
            else:
                retry_signals["fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message"] = 1
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message: no mixed user message or split boundary detected in prepared payload"
                )
            logger.warning(
                "fail_open_retry_upstream_pairing_disagreement: upstream reported pairing failure while local strict validation passed (prepared_mixed_user_tool_result_messages=%s prepared_split_boundaries=%s)",
                prepared_mixed_user_tool_result_messages,
                prepared_split_boundaries,
            )
            session.capture_event(
                {
                    "event": "fail_open_retry_upstream_pairing_disagreement",
                    "error_text": error_text[:500],
                    "prepared_summary": prepared_summary,
                    "prepared_pairing": prepared_pairing,
                    "provider_safe_summary": provider_safe_summary,
                    "provider_safe_pairing": provider_safe_pairing,
                    "behavior_signals": retry_signals,
                }
            )
        if fallback_content is not None and _payloads_materially_differ(content, fallback_content):
            provider_safe_outgoing_failures = (
                validate_anthropic_outgoing_bridge_body(fallback_body)
                if isinstance(fallback_body, dict)
                else ["body_not_dict"]
            )
            if has_blocking_outgoing_failures(provider_safe_failures + provider_safe_outgoing_failures):
                retry_signals["fail_open_retry_provider_safe_invalid"] = 1
                retry_candidate_valid = False
                if (
                    allow_original_retry
                    and original_content is not None
                    and _payloads_materially_differ(fallback_content, original_content)
                ):
                    original_outgoing_failures = _validate_outgoing_bridge_body_from_bytes(original_content)
                    original_body = _decode_bridge_body(original_content)
                    original_bridge_failures = (
                        validate_anthropic_bridge_body(original_body)
                        if isinstance(original_body, dict)
                        else ["body_not_dict"]
                    )
                    original_has_hard_failures = (
                        original_bridge_failures
                        and not has_recoverable_immediate_pairing_failures(original_bridge_failures)
                    )
                    if not original_outgoing_failures and not original_has_hard_failures:
                        retry_candidate_content = original_content
                        retry_candidate_kind = "original"
                        retry_candidate_valid = True
                        retry_signals["fail_open_retry_original_after_provider_safe_invalid"] = 1
                        logger.warning(
                            "fail_open_retry_original_after_provider_safe_invalid: provider-safe payload failed final local validation; retrying with raw original payload"
                        )
                        session.capture_event(
                            {
                                "event": "fail_open_retry_original_after_provider_safe_invalid",
                                "strict_failures": provider_safe_failures,
                                "outgoing_failures": provider_safe_outgoing_failures,
                                "original_outgoing_failures": original_outgoing_failures,
                                "original_bridge_failures": original_bridge_failures,
                                "prepared_summary": prepared_summary,
                                "prepared_pairing": prepared_pairing,
                                "provider_safe_summary": provider_safe_summary,
                                "provider_safe_pairing": provider_safe_pairing,
                                "behavior_signals": retry_signals,
                            }
                        )
                    else:
                        logger.warning(
                            "fail_open_retry_provider_safe_invalid: refusing retry because provider-safe payload failed final local validation and raw original payload also failed validation: strict=%s outgoing=%s original_outgoing=%s original_bridge=%s",
                            provider_safe_failures,
                            provider_safe_outgoing_failures,
                            original_outgoing_failures,
                            original_bridge_failures,
                        )
                        if has_provider_sensitive_failures(
                            provider_safe_outgoing_failures
                        ) or has_provider_sensitive_failures(original_outgoing_failures):
                            retry_signals["fail_open_retry_provider_safe_blocked_local"] = 1
                            retry_signals["tok_bridge_provider_pairing_risk_detected"] = 1
                        session.capture_event(
                            {
                                "event": "fail_open_retry_provider_safe_invalid",
                                "strict_failures": provider_safe_failures,
                                "outgoing_failures": provider_safe_outgoing_failures,
                                "original_outgoing_failures": original_outgoing_failures,
                                "original_bridge_failures": original_bridge_failures,
                                "prepared_summary": prepared_summary,
                                "prepared_pairing": prepared_pairing,
                                "provider_safe_summary": provider_safe_summary,
                                "provider_safe_pairing": provider_safe_pairing,
                                "behavior_signals": retry_signals,
                            }
                        )
                        return response, retried_without_tok, retry_signals
                else:
                    logger.warning(
                        "fail_open_retry_provider_safe_invalid: refusing retry because provider-safe payload failed final local validation: strict=%s outgoing=%s",
                        provider_safe_failures,
                        provider_safe_outgoing_failures,
                    )
                    if has_provider_sensitive_failures(provider_safe_outgoing_failures):
                        retry_signals["fail_open_retry_provider_safe_blocked_local"] = 1
                        retry_signals["tok_bridge_provider_pairing_risk_detected"] = 1
                    session.capture_event(
                        {
                            "event": "fail_open_retry_provider_safe_invalid",
                            "strict_failures": provider_safe_failures,
                            "outgoing_failures": provider_safe_outgoing_failures,
                            "prepared_summary": prepared_summary,
                            "prepared_pairing": prepared_pairing,
                            "provider_safe_summary": provider_safe_summary,
                            "provider_safe_pairing": provider_safe_pairing,
                            "behavior_signals": retry_signals,
                        }
                    )
                    return response, retried_without_tok, retry_signals
            if retry_candidate_valid and retry_candidate_kind == "provider-safe":
                retry_signals["fail_open_retry_provider_safe_validated"] = 1
                logger.info(
                    "fail_open_retry_provider_safe_validated: %s fallback passed local strict validation",
                    retry_candidate_kind,
                )
        if retry_candidate_content is not None and _payloads_materially_differ(content, retry_candidate_content):
            retry_kind = retry_candidate_kind
            logger.warning(
                "Upstream 400 after Tok request preparation: %s; retrying with %s payload",
                error_text[:500],
                retry_kind,
            )
            await response.aclose()
            request_obj = client.build_request(method, url, headers=headers, content=retry_candidate_content)
            response = await client.send(request_obj, stream=stream)
            retried_without_tok = True
            retry_signals["fail_open_retry_usage"] = 1
            if retry_kind == "provider-safe":
                retry_signals["fail_open_retry_provider_safe"] = 1
            else:
                retry_signals["fail_open_retry_original_after_provider_safe_invalid"] = retry_signals.get(
                    "fail_open_retry_original_after_provider_safe_invalid", 0
                )
            if not allow_original_retry:
                retry_signals["fail_open_raw_retry_blocked"] = 1
        elif not allow_original_retry and original_content is not None:
            logger.warning(
                "fail_open_raw_retry_blocked: refusing to resend raw original payload after tool-history repair"
            )
            retry_signals["fail_open_raw_retry_blocked"] = 1
    return response, retried_without_tok, retry_signals
