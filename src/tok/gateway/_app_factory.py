"""Gateway app-construction helpers behind the public interface."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import math
import random  # noqa: F401 - compatibility anchor for gateway tests
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from tok.runtime.pipeline.request_validation import normalize_tool_use_blocks
from tok.runtime.smoothness import SmoothnessEventType

from . import (
    _RUNTIME,
    ANTHROPIC_API_BASE,
    BridgeSession,
    _record_fallback_once,
    _request_policy_mode_label,
    logger,
)
from ._anthropic_optimizations import apply_anthropic_optimizations
from ._bridge_comparison import _safe_headers
from ._bridge_request_handler import send_with_tok_fail_open_retry
from ._bridge_runtime_pipeline import prepare_bridge_payload
from ._bridge_streaming import _emit_sse_block, buffer_strip_restream_impl

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["buffer_strip_restream_impl", "create_app_impl"]


def _upstream_host(api_base: str) -> str:
    parsed = urlsplit(api_base.strip() or ANTHROPIC_API_BASE)
    if parsed.netloc:
        return parsed.netloc
    fallback = parsed.path.split("/", 1)[0].strip()
    return fallback or "api.anthropic.com"


def _upstream_target_url(api_base: str, path: str, query: str) -> str:
    base = api_base.strip() or ANTHROPIC_API_BASE
    target_url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    if query:
        target_url += f"?{query}"
    return target_url


async def _aclose_if_possible(resource: object | None) -> None:
    if resource is None:
        return
    resource_aclose = getattr(resource, "aclose", None)
    if callable(resource_aclose):
        await resource_aclose()  # type: ignore[misc]


async def _close_streaming_setup_resources(
    response: object | None,
    client: object | None,
) -> None:
    await _aclose_if_possible(response)
    await _aclose_if_possible(client)


def _note_request_policy_recovery_watch(session: BridgeSession, signals: dict[str, int] | None) -> None:
    if not signals:
        return
    if signals.get("stream_recovery_retry", 0) or signals.get("stream_recovery_fallback", 0):
        session.runtime_session.note_request_policy_stream_recovery()
    if (
        signals.get("fail_open_retry_upstream_pairing_disagreement", 0)
        or signals.get("tok_bridge_provider_pairing_risk_detected", 0)
        or signals.get("tok_bridge_pairing_degraded_to_provider_safe", 0)
        or signals.get("tok_bridge_assistant_tool_use_text_interleaving_blocked", 0)
        or signals.get("tok_bridge_invalid_tool_history_recovery", 0)
        or signals.get("tok_bridge_invalid_tool_history_quarantined", 0)
        or signals.get("tok_bridge_invalid_tool_history_blocked", 0)
        or signals.get("tok_history_pairing_safety_degraded", 0)
    ):
        session.runtime_session.note_request_policy_tool_mode_recovery()


def _rate_limit_throttle_remaining(session: BridgeSession) -> float:
    remaining = session._rate_limit_throttle_until - time.time()
    return max(0.0, remaining)


def _is_rate_limited(session: BridgeSession) -> bool:
    return _rate_limit_throttle_remaining(session) > 0.0


def _record_rate_limit_hit(session: BridgeSession) -> None:
    now = time.time()
    history = session._rate_limit_429_history

    window_sec = max(1, int(session.rate_limit_throttle_window_sec))
    history[:] = [ts for ts in history if now - ts <= window_sec]
    history.append(now)

    if (
        session.rate_limit_throttle_threshold > 0
        and len(history) >= session.rate_limit_throttle_threshold
        and session.rate_limit_throttle_cooldown_sec > 0
    ):
        session._rate_limit_throttle_until = now + float(session.rate_limit_throttle_cooldown_sec)


def _build_rate_limit_response(retry_after_seconds: float) -> Response:
    retry_after_header = str(max(1, math.ceil(retry_after_seconds)))
    return Response(
        content=json.dumps(
            {
                "error": {
                    "type": "rate_limit_error",
                    "message": "Tok gateway is temporarily rate limited; retry later.",
                }
            }
        ),
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": retry_after_header},
    )


def _normalize_rate_limit_response(session: BridgeSession, response: httpx.Response) -> Response:
    _record_rate_limit_hit(session)
    remaining = _rate_limit_throttle_remaining(session)
    retry_after_header = response.headers.get("retry-after")
    retry_after_seconds = remaining
    if retry_after_seconds <= 0.0 and retry_after_header:
        try:
            retry_after_seconds = max(0.0, float(retry_after_header))
        except ValueError:
            retry_after_seconds = 1.0
    if retry_after_seconds <= 0.0:
        retry_after_seconds = 1.0
    return _build_rate_limit_response(retry_after_seconds)


def _rebuild_content_preserving_position(
    original_content: list[dict[str, Any]],
    processed_blocks: list[dict[str, Any]],
    passthrough_blocks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Replace text blocks with processed output while keeping non-text blocks in place.

    Non-text blocks are only replaced when they carry a unique ``id`` field that
    matches a passthrough entry (e.g. normalised tool_use blocks).  Blocks
    without an ``id`` — thinking, redacted_thinking — are always emitted
    verbatim from *original_content* so their cryptographic signatures are
    never altered.
    """
    passthrough_by_id: dict[str, dict[str, Any]] = {}
    if passthrough_blocks is not None:
        for block in passthrough_blocks:
            if isinstance(block, dict):
                block_id = block.get("id", "")
                if block_id:
                    passthrough_by_id[block_id] = block
    result: list[dict[str, Any]] = []
    processed_idx = 0
    for block in original_content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            if processed_idx < len(processed_blocks):
                result.append(processed_blocks[processed_idx])
                processed_idx += 1
            else:
                result.append(block)
        else:
            block_id = block.get("id", "")
            if block_id and block_id in passthrough_by_id:
                result.append(passthrough_by_id[block_id])
            else:
                result.append(block)
    while processed_idx < len(processed_blocks):
        result.append(processed_blocks[processed_idx])
        processed_idx += 1
    return result


async def _json_to_sse(resp_json: dict[str, Any]) -> AsyncIterator[bytes]:
    model = resp_json.get("model", "")
    usage = resp_json.get("usage", {})
    message_start = {
        "type": "message_start",
        "message": {"model": model, "usage": usage},
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()
    content = resp_json.get("content", [])
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        for event_bytes in _emit_sse_block(i, block):
            yield event_bytes
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": resp_json.get("stop_reason", "end_turn")},
        "usage": usage,
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()
    yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def create_app_impl(session: BridgeSession | None = None) -> FastAPI:
    """Create the bridge FastAPI application."""
    if session is None:
        session = BridgeSession()

    app = FastAPI(title="tok-bridge")

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root() -> dict[str, str]:
        return {"status": "ok", "bridge": "tok"}

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health() -> dict[str, Any]:
        session_summary = session.tracker.session_summary() or {}
        signals = dict(session.tracker.behavior_signals())
        for (
            key,
            value,
        ) in session.runtime_session.pending_behavior_signals.items():
            signals[key] = signals.get(key, 0) + int(value)
        return {
            "status": "ok",
            "bridge": "tok",
            "port": session.port,
            "api_base": session.api_base,
            "mode": _request_policy_mode_label(session.request_policy_default),
            "request_policy": session.request_policy_default,
            "baseline_only": session.runtime_session._baseline_only,
            "persistence_failures": session.runtime_session._persistence_failures,
            "fallback_count": int(
                session_summary.get(
                    "fallback_count",
                    session.tracker.behavior_signals().get("tok_fallback_activated", 0),
                )
            ),
            "actual_tokens": int(session_summary.get("actual_tokens", 0)),
            "baseline_tokens": int(session_summary.get("baseline_tokens", 0)),
            "session_tokens_saved": int(session_summary.get("tokens_saved", 0)),
            "baseline_prompt_tokens": int(session_summary.get("baseline_prompt_tokens", 0)),
            "prepared_prompt_tokens": int(session_summary.get("prepared_prompt_tokens", 0)),
            "saved_prompt_tokens": int(session_summary.get("saved_prompt_tokens", 0)),
            "session_savings_pct": float(session_summary.get("savings_pct", 0.0)),
            "actual_cost_usd": float(session_summary.get("actual_cost_usd", 0.0)),
            "baseline_cost_usd": float(session_summary.get("baseline_cost_usd", 0.0)),
            "cost_saved_usd": float(session_summary.get("cost_saved_usd", 0.0)),
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
            "non_tok_count": int(session_summary.get("non_tok_count", signals.get("non_tok_response", 0))),
            "answer_anchor_miss_count": int(session_summary.get("answer_anchor_miss_count", 0)),
            "repeat_search_count": int(signals.get("repeat_search", 0)),
            "repeat_file_read_count": int(signals.get("repeat_file_read", 0)),
            "shell_file_read_normalized_count": int(signals.get("shell_file_read_normalized", 0)),
            "shell_file_snapshot_captured_count": int(signals.get("shell_file_snapshot_captured", 0)),
            "repeat_target_hot_count": int(signals.get("repeat_target_hot", 0)),
            "repeat_target_stuck_count": int(signals.get("repeat_target_stuck", 0)),
            "hot_recent_hint_count": int(signals.get("hot_recent_hint_injected", 0)),
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
            "state_resend_full_count": int(signals.get("state_resend_full_turn", 0)),
            "state_resend_delta_count": int(signals.get("state_resend_delta_turn", 0)),
            "state_resend_suppressed_count": int(signals.get("state_resend_suppressed_turn", 0)),
            "stream_recovery_attempt_count": int(session_summary.get("stream_recovery_attempt_count", 0)),
            "stream_recovery_success_text_count": int(session_summary.get("stream_recovery_success_text_count", 0)),
            "stream_recovery_success_tool_use_count": int(
                session_summary.get("stream_recovery_success_tool_use_count", 0)
            ),
            "stream_recovery_fallback_count": int(session_summary.get("stream_recovery_fallback_count", 0)),
            "stream_recovery_empty_success_count": int(
                session_summary.get(
                    "stream_recovery_empty_success_count",
                    signals.get("stream_recovery_empty_success", 0),
                )
            ),
            "stream_recovery_read_error_count": int(
                session_summary.get(
                    "stream_recovery_read_error_count",
                    signals.get("stream_recovery_read_error", 0),
                )
            ),
            "tool_history_repaired_count": int(session_summary.get("tool_history_repaired_count", 0)),
            "tool_history_pairing_repaired_count": int(session_summary.get("tool_history_pairing_repaired_count", 0)),
            "tool_history_quarantined_count": int(session_summary.get("tool_history_quarantined_count", 0)),
            "tool_history_blocked_count": int(session_summary.get("tool_history_blocked_count", 0)),
            "invalid_tool_history_session_reset_count": int(
                session_summary.get("invalid_tool_history_session_reset_count", 0)
            ),
            "provider_pairing_disagreement_count": int(session_summary.get("provider_pairing_disagreement_count", 0)),
            "assistant_tool_use_text_interleaving_blocked_count": int(
                session_summary.get(
                    "assistant_tool_use_text_interleaving_blocked_count",
                    signals.get(
                        "tok_bridge_assistant_tool_use_text_interleaving_blocked",
                        0,
                    ),
                )
            ),
            "preflight_block_original_payload_count": int(
                session_summary.get(
                    "preflight_block_original_payload_count",
                    signals.get("preflight_block_original_payload", 0),
                )
            ),
            "preflight_block_rewritten_payload_count": int(
                session_summary.get(
                    "preflight_block_rewritten_payload_count",
                    signals.get("preflight_block_rewritten_payload", 0),
                )
            ),
            "request_policy_natural_first_count": int(session_summary.get("request_policy_natural_first_count", 0)),
            "request_policy_tool_compatible_count": int(session_summary.get("request_policy_tool_compatible_count", 0)),
            "request_policy_escalations_count": int(session_summary.get("request_policy_escalations_count", 0)),
            "request_policy_deescalations_count": int(session_summary.get("request_policy_deescalations_count", 0)),
            "request_policy_interleaving_downgrades_count": int(
                session_summary.get("request_policy_interleaving_downgrades_count", 0)
            ),
            "request_policy_reason_stream_recovery_count": int(
                session_summary.get(
                    "request_policy_reason_stream_recovery_count",
                    signals.get("request_policy_reason_stream_recovery", 0),
                )
            ),
            "request_policy_reason_tool_recovery_count": int(
                session_summary.get(
                    "request_policy_reason_tool_recovery_count",
                    signals.get("request_policy_reason_tool_recovery", 0),
                )
            ),
            "request_policy_reason_structured_tool_loop_count": int(
                session_summary.get(
                    "request_policy_reason_structured_tool_loop_count",
                    signals.get("request_policy_reason_structured_tool_loop", 0),
                )
            ),
            "request_policy_held_by_recovery_count": int(
                session_summary.get(
                    "request_policy_held_by_recovery_count",
                    signals.get("request_policy_held_by_recovery", 0)
                    + signals.get("request_policy_recovery_sticky_continuations", 0),
                )
            ),
            "session_quality": str(session_summary.get("session_quality", "clean")),
            "last_degradation_reason": str(session_summary.get("last_degradation_reason", "")),
            "smoothness_score": session.runtime_session.latest_turn_smoothness_score,
            "labour_index": session.runtime_session.latest_turn_labour_index,
            "current_mode": session.runtime_session.current_tok_mode.name,
            "stream_instability_events": sum(
                v for k, v in session.runtime_session.smoothness_event_counts.items() if "stream" in k.lower()
            ),
            "thinking_mutation_events": int(
                session.runtime_session.smoothness_event_counts.get("thinking_block_mutation", 0)
            ),
            "task_score": session.runtime_session.current_task_smoothness_score,
            "repeated_active_file_reads": int(signals.get("repeat_file_read", 0)),
        }

    @app.post("/reset-session")
    async def reset_session_endpoint() -> dict[str, str]:
        """Reset per-session first-read / first-exact state for a new conversation."""
        session.tracker.reset_session_stats()
        session.runtime_session.reset_session()
        return {"status": "ok", "action": "session_reset"}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def bridge(request: Request, path: str) -> Response:
        body_bytes = await request.body()
        original_body_bytes = body_bytes
        request_state = {"fallback_recorded": False}

        if path.startswith("v1/") and _is_rate_limited(session):
            logger.warning(
                "rate_limit_throttle_active: blocking request to %s for %.2fs",
                path,
                _rate_limit_throttle_remaining(session),
            )
            return _build_rate_limit_response(_rate_limit_throttle_remaining(session))

        session.smoothness_tracker.start_turn()

        skip = {
            "host",
            "content-length",
            "accept-encoding",
            "connection",
            "x-tok-tool-compatible",
        }
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        headers["host"] = _upstream_host(session.api_base)

        has_x_api_key = any(k.lower() == "x-api-key" for k in headers)
        has_auth_bearer = any(
            k.lower() == "authorization" and v.lower().startswith("bearer ") for k, v in headers.items()
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
        request_policy = "forced_baseline"
        provider_safe_original_body_bytes = original_body_bytes
        raw_retry_forbidden = False

        if path in {"v1/messages", "v1/messages/count_tokens"} and request.method == "POST" and body_bytes:
            try:
                body = json.loads(body_bytes)
                original_body = copy.deepcopy(body) if isinstance(body, dict) else body
                if not isinstance(body, dict) or not isinstance(original_body, dict):
                    msg = "request body must be a JSON object"
                    raise ValueError(msg)
                tok_tool_header = request.headers.get("x-tok-tool-compatible", "")
                bridge_payload, preflight_response = prepare_bridge_payload(
                    session=session,
                    body=body,
                    headers=headers,
                    path=path,
                    tok_tool_header=tok_tool_header,
                    request_state=request_state,
                )
                if preflight_response is not None:
                    return preflight_response

                body = dict(bridge_payload.body)
                behavior_signals = dict(bridge_payload.behavior_signals)
                request_policy = bridge_payload.request_policy
                request_tool_compatible = bridge_payload.request_tool_compatible
                compressed = bridge_payload.compressed
                saved_toks = bridge_payload.saved_toks
                tool_breakdown = dict(bridge_payload.tool_breakdown)
                prompt_metrics = dict(bridge_payload.prompt_metrics)
                raw_retry_forbidden = bridge_payload.retry_forbidden
                provider_safe_original_body = dict(bridge_payload.provider_safe_original_body)
                provider_safe_original_body_bytes = json.dumps(provider_safe_original_body).encode()
                request_model = bridge_payload.request_model
                messages = list(bridge_payload.request_messages)

                if path == "v1/messages":
                    from tok.runtime.smoothness.models import TokMode

                    if session.runtime_session.current_tok_mode == TokMode.LOSSLESS_TASK_MODE:
                        logger.warning("LOSSLESS_TASK_MODE active: preserving full task flow for emergency mode")

                    _note_request_policy_recovery_watch(session, behavior_signals)
                    if behavior_signals.get("smoothness_history_winnowing_active_loop"):
                        with contextlib.suppress(Exception):
                            session.smoothness_tracker.record(
                                SmoothnessEventType.HISTORY_WINNOWING_ACTIVE_LOOP,
                            )
                    if behavior_signals.get("smoothness_prompt_optimization_active_task"):
                        with contextlib.suppress(Exception):
                            session.smoothness_tracker.record(
                                SmoothnessEventType.PROMPT_OPTIMIZATION_ACTIVE_TASK,
                            )
                    if behavior_signals.get("repeat_file_read", 0) > 0:
                        with contextlib.suppress(Exception):
                            session.smoothness_tracker.record(
                                SmoothnessEventType.REPEATED_ACTIVE_FILE_READ,
                                {"count": behavior_signals.get("repeat_file_read", 0)},
                            )

                    session.capture_request(
                        {
                            "event": "request",
                            "messages": messages,
                            "system": body.get("system", ""),
                            "model": request_model,
                            "tool_compatible": request_tool_compatible,
                            "request_policy": request_policy,
                        }
                    )
                    body = apply_anthropic_optimizations(body)
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
                            "request_policy": "forced_baseline",
                        }
                    )

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
                if session.fail_open:
                    logger.error(
                        "tok_fallback_activated: critical system error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["critical_system_error"] = 1
                    _record_fallback_once(session, request_state)
                else:
                    raise
            except Exception as exc:
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
                    logger.error("Unexpected error in request processing: %s", exc)
                    raise

        target_url = _upstream_target_url(session.api_base, path, request.url.query)

        is_streaming = False
        thinking_forced_non_stream = False
        try:
            body_dict = json.loads(body_bytes)
            is_streaming = bool(body_dict.get("stream", False))
            thinking_cfg = body_dict.get("thinking") if isinstance(body_dict, dict) else None
            if is_streaming and isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled":
                body_dict["stream"] = False
                body_bytes = json.dumps(body_dict).encode()
                is_streaming = False
                thinking_forced_non_stream = True
                behavior_signals["thinking_forced_non_stream"] = 1
                logger.info("thinking_forced_non_stream: extended thinking detected, forcing non-streaming upstream")
            if is_streaming:
                from tok.runtime.smoothness.models import TokMode

                current_mode = session.runtime_session.current_tok_mode
                if current_mode in (
                    TokMode.SMOOTH_MODE,
                    TokMode.LOSSLESS_TASK_MODE,
                ):
                    body_dict["stream"] = False
                    body_bytes = json.dumps(body_dict).encode()
                    is_streaming = False
                    behavior_signals["smoothness_streaming_disabled"] = 1
                    logger.info(
                        "smoothness_streaming_disabled: SMOOTH_MODE or LOSSLESS_TASK_MODE active, forcing non-streaming (mode=%s)",
                        current_mode.value,
                    )
        except (json.JSONDecodeError, ValueError):
            logger.debug("Failed to parse JSON for streaming detection")
        except Exception as exc:
            logger.debug("Unexpected error during JSON parsing: %s", exc)

        if is_streaming:
            client = httpx.AsyncClient(timeout=300.0)
            try:
                (
                    response,
                    retried_without_tok,
                    retry_signals,
                ) = await send_with_tok_fail_open_retry(
                    session,
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
                    sleep_fn=asyncio.sleep,
                )
                if retry_signals:
                    for key, value in retry_signals.items():
                        behavior_signals[key] = behavior_signals.get(key, 0) + value
                _note_request_policy_recovery_watch(session, retry_signals)
                if response.status_code == 429:
                    await _close_streaming_setup_resources(response, client)
                    return _normalize_rate_limit_response(session, response)
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
                    behavior_signals["tok_fail_open_retry"] = behavior_signals.get("tok_fail_open_retry", 0) + 1
                    behavior_signals["tok_fallback_activated"] = behavior_signals.get("tok_fallback_activated", 0) + 1
                    logger.warning("tok_fallback_activated: upstream 400 retry, serving without compression")
                    _record_fallback_once(session, request_state)
                resp_headers = _safe_headers(response.headers)
                if response.status_code >= 400:
                    error_content = await response.aread()
                    await _close_streaming_setup_resources(None, client)
                    return Response(
                        content=error_content,
                        status_code=response.status_code,
                        headers=resp_headers,
                        media_type=response.headers.get("content-type", "application/json"),
                    )

                if path == "v1/messages":
                    # Ownership transfers here. The streaming implementation must close both response and client.
                    return StreamingResponse(
                        buffer_strip_restream_impl(
                            session,
                            client,
                            response,
                            input_saved_tokens=saved_toks if compressed else 0,
                            type_breakdown=tool_breakdown if compressed else None,
                            behavior_signals=behavior_signals or None,
                            prompt_metrics=prompt_metrics if compressed else None,
                            tool_compatible=request_tool_compatible,
                            request_method=request.method,
                            request_url=target_url,
                            request_headers=headers,
                            request_content=(provider_safe_original_body_bytes if retried_without_tok else body_bytes),
                            request_state=request_state,
                            client_owned=True,
                        ),
                        status_code=response.status_code,
                        headers=resp_headers,
                        media_type=response.headers.get("content-type", "text/event-stream"),
                    )
            except Exception as e:
                logger.error(
                    "Streaming error in Tok bridge: %s",
                    str(e),
                    exc_info=True,
                )
                await _close_streaming_setup_resources(None, client)
                if session.fail_open:
                    logger.warning("Streaming error - fail-open: retrying without Tok")
                    compressed = False
                    saved_toks = 0
                    tool_breakdown = {}
                    behavior_signals["streaming_error_retry"] = behavior_signals.get("streaming_error_retry", 0) + 1
                    retry_client = httpx.AsyncClient(timeout=300.0)
                    response = None
                    try:
                        (
                            response,
                            _retried,
                            retry_signals,
                        ) = await send_with_tok_fail_open_retry(
                            session,
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
                        for key, value in retry_signals.items():
                            behavior_signals[key] = behavior_signals.get(key, 0) + value
                        _note_request_policy_recovery_watch(session, retry_signals)
                        if path == "v1/messages":
                            # Ownership transfers here. The streaming implementation must close both response and client.
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
                                    client_owned=True,
                                ),
                                status_code=response.status_code,
                                headers=_safe_headers(response.headers),
                                media_type=response.headers.get("content-type", "text/event-stream"),
                            )
                    except Exception:
                        await _close_streaming_setup_resources(response, retry_client)
                        raise
                return Response(
                    content=f"Streaming error: {e!s}",
                    status_code=502,
                    media_type="text/plain",
                )

        async with httpx.AsyncClient(timeout=300.0) as client:
            (
                response,
                retried_without_tok,
                retry_signals,
            ) = await send_with_tok_fail_open_retry(
                session,
                client,
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes,
                original_content=original_body_bytes,
                retry_content=provider_safe_original_body_bytes,
                allow_original_retry=not raw_retry_forbidden,
                compressed_request=compressed,
                sleep_fn=asyncio.sleep,
            )
            if retry_signals:
                for key, value in retry_signals.items():
                    behavior_signals[key] = behavior_signals.get(key, 0) + value
            _note_request_policy_recovery_watch(session, retry_signals)
            if response.status_code == 429:
                return _normalize_rate_limit_response(session, response)
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
                behavior_signals["tok_fail_open_retry"] = behavior_signals.get("tok_fail_open_retry", 0) + 1
                behavior_signals["tok_fallback_activated"] = behavior_signals.get("tok_fallback_activated", 0) + 1
                logger.warning("tok_fallback_activated: upstream 400 retry, serving without compression")
                _record_fallback_once(session, request_state)

            content = response.content

            if response.status_code >= 400:
                logger.warning(
                    "Upstream %d: %s",
                    response.status_code,
                    content[:300].decode(errors="replace") if isinstance(content, bytes) else str(content)[:300],
                )
            resp_json: dict[str, Any] = {}
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
                    (
                        passthrough_blocks,
                        passthrough_signals,
                    ) = normalize_tool_use_blocks(passthrough_blocks, seed_prefix="toolu_upstream")
                    for key, value in passthrough_signals.items():
                        behavior_signals[key] = behavior_signals.get(key, 0) + value

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
                        new_content = _rebuild_content_preserving_position(
                            resp_json.get("content", []),
                            processed.content_blocks,
                            passthrough_blocks,
                        )
                        resp_json["content"] = new_content
                        total_output_saved = processed.output_saved_tokens

                        session_signals = session.consume_behavior_signals()
                        if session_signals:
                            response_signals = response_signals or {}
                            for k, v in session_signals.items():
                                response_signals[k] = response_signals.get(k, 0) + v

                        logger.info(
                            "Response: %d blocks, ~%d saved",
                            len(new_content),
                            total_output_saved,
                        )
                    else:
                        session_signals = session.runtime_session.consume_behavior_signals()
                        if session_signals:
                            response_signals = dict(session_signals)

                    _note_request_policy_recovery_watch(session, response_signals)

                    usage = resp_json.get("usage", {})
                    if resp_json.get("model") and usage:
                        session.tracker.record_call(
                            model=resp_json["model"],
                            actual_input=usage.get("input_tokens", 0),
                            actual_output=usage.get("output_tokens", 0),
                            cache_read=usage.get("cache_read_input_tokens", 0),
                            cache_write=usage.get("cache_creation_input_tokens", 0),
                            input_saved=saved_toks if compressed else 0,
                            output_saved=total_output_saved,
                            type_breakdown=tool_breakdown if compressed else None,
                            behavior_signals=response_signals or None,
                            prompt_metrics=prompt_metrics if compressed else None,
                        )

                        # Finish smoothness turn
                        turn_report = session.smoothness_tracker.finish_turn()

                        # Update session state with smoothness data
                        event_counts: dict[str, int] = {}
                        for event in turn_report.events:
                            key = event.event_type.value
                            event_counts[key] = event_counts.get(key, 0) + 1

                        session.runtime_session.update_smoothness_state(
                            turn_score=turn_report.score,
                            labour_index=turn_report.labour_index,
                            tok_mode=turn_report.mode,
                            event_counts=event_counts,
                        )

                        logger.info(
                            "Smoothness turn complete | turn_id=%s | score=%d | labour_index=%d | mode=%s | events=%d",
                            turn_report.turn_id,
                            turn_report.score,
                            turn_report.labour_index,
                            turn_report.mode.value,
                            len(turn_report.events),
                        )
                        session.capture_event(
                            {
                                "event": "response",
                                "model": resp_json["model"],
                                "request_policy": request_policy,
                                "tool_compatible": request_tool_compatible,
                                "baseline_only": session.runtime_session._baseline_only,
                                "persistence_failures": session.runtime_session._persistence_failures,
                                "fallback_count": int(
                                    session.tracker.behavior_signals().get("tok_fallback_activated", 0)
                                ),
                                "behavior_signals": response_signals or {},
                                "session_quality": str(
                                    (session.tracker.session_summary() or {}).get("session_quality", "clean")
                                ),
                                "session_tokens_saved": int(
                                    (session.tracker.session_summary() or {}).get("tokens_saved", 0)
                                ),
                                "session_savings_pct": float(
                                    (session.tracker.session_summary() or {}).get("savings_pct", 0.0)
                                ),
                                "last_degradation_reason": str(
                                    (session.tracker.session_summary() or {}).get("last_degradation_reason", "")
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
                            session_signals = session.consume_behavior_signals()
                            error_signals = {"processing_error": 1}
                            if session_signals:
                                for k, v in session_signals.items():
                                    error_signals[k] = error_signals.get(k, 0) + v

                            if _model and _usage:
                                session.tracker.record_call(
                                    model=_model,
                                    actual_input=_usage.get("input_tokens", 0),
                                    actual_output=_usage.get("output_tokens", 0),
                                    cache_read=_usage.get("cache_read_input_tokens", 0),
                                    cache_write=_usage.get("cache_creation_input_tokens", 0),
                                    input_saved=saved_toks if compressed else 0,
                                    output_saved=0,
                                    type_breakdown=tool_breakdown if compressed else None,
                                    behavior_signals=error_signals,
                                    prompt_metrics=prompt_metrics if compressed else None,
                                )
                        except Exception as _exc:
                            logger.debug(
                                "Failed to record usage in fail-open path: %s",
                                _exc,
                            )
                    else:
                        raise

            if thinking_forced_non_stream and response.status_code == 200 and resp_json:
                sse_headers = _safe_headers(response.headers)
                sse_headers["content-type"] = "text/event-stream"
                return StreamingResponse(
                    _json_to_sse(resp_json),
                    status_code=200,
                    headers=sse_headers,
                    media_type="text/event-stream",
                )

            return Response(
                content=content,
                status_code=response.status_code,
                headers=_safe_headers(response.headers),
                media_type=response.headers.get("content-type", "application/json"),
            )

    return app
