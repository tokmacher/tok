"""Gateway app-construction helpers behind the public interface."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import json
import math
import random  # noqa: F401 - compatibility anchor for gateway tests
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from tok.runtime._diagnostics import DiagnosticsSnapshot
from tok.runtime.pipeline.request_validation import normalize_tool_use_blocks
from tok.runtime.smoothness import SmoothnessEventType
from tok.spec.live_trace import emit_live_trace

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
from ._bridge_streaming import _emit_sse_block, _run_macro_mining, buffer_strip_restream_impl, passthrough_stream_impl
from ._types import build_capability_manifest

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


def _handle_retried_without_tok(
    behavior_signals: dict[str, int],
    active_session: BridgeSession,
    request_state: dict[str, bool],
) -> tuple[bool, int, dict[str, int], dict[str, int]]:
    compressed = False
    saved_toks = 0
    tool_breakdown: dict[str, int] = {}
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
    _record_fallback_once(active_session, request_state)
    return compressed, saved_toks, tool_breakdown, prompt_metrics


def _expand_macros_in_blocks(blocks: list[dict[str, Any]], active_session: BridgeSession) -> list[dict[str, Any]]:
    from tok.macros.expansion import expand_tool_use_blocks

    registry = active_session.runtime_session.bridge_memory.macro_registry
    if not registry.macros:
        return blocks
    return expand_tool_use_blocks(blocks, registry)


def _build_response_signals(
    full_response_text: str,
    resp_json: dict[str, Any],
    active_session: BridgeSession,
    behavior_signals: dict[str, int],
    request_tool_compatible: bool,
) -> tuple[dict[str, int], int]:
    total_output_saved = 0
    response_signals: dict[str, int] = {}

    passthrough_blocks = [
        block for block in resp_json.get("content", []) if isinstance(block, dict) and block.get("type") != "text"
    ]
    passthrough_blocks, passthrough_signals = normalize_tool_use_blocks(
        passthrough_blocks, seed_prefix="toolu_upstream"
    )
    passthrough_blocks = _expand_macros_in_blocks(passthrough_blocks, active_session)
    for key, value in passthrough_signals.items():
        behavior_signals[key] = behavior_signals.get(key, 0) + value

    if full_response_text:
        processed = _RUNTIME.process_response(
            full_response_text,
            model=str(resp_json.get("model", "")),
            session=active_session.runtime_session,
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

        session_signals = active_session.runtime_session.consume_behavior_signals()
        if session_signals:
            for k, v in session_signals.items():
                response_signals[k] = response_signals.get(k, 0) + v

        logger.info(
            "Response: %d blocks, ~%d saved",
            len(new_content),
            total_output_saved,
        )
    else:
        passthrough_idx = 0
        for i, block in enumerate(resp_json.get("content", [])):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                continue
            if passthrough_idx < len(passthrough_blocks):
                resp_json["content"][i] = passthrough_blocks[passthrough_idx]
                passthrough_idx += 1
        while passthrough_idx < len(passthrough_blocks):
            resp_json["content"].append(passthrough_blocks[passthrough_idx])
            passthrough_idx += 1

        session_signals = active_session.runtime_session.consume_behavior_signals()
        if session_signals:
            for k, v in session_signals.items():
                response_signals[k] = response_signals.get(k, 0) + v

    _note_request_policy_recovery_watch(active_session, response_signals)
    return response_signals, total_output_saved


def _rebuild_and_record_response(
    resp_json: dict[str, Any],
    active_session: BridgeSession,
    saved_toks: int,
    compressed: bool,
    tool_breakdown: dict[str, int],
    response_signals: dict[str, int],
    prompt_metrics: dict[str, int],
    total_output_saved: int,
    request_policy: str,
    request_tool_compatible: bool,
) -> None:
    usage = resp_json.get("usage", {})
    if not (resp_json.get("model") and usage):
        return

    active_session.tracker.record_call(
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
    try:
        asyncio.get_running_loop().create_task(_run_macro_mining(active_session))
    except RuntimeError:
        pass

    turn_report = active_session.smoothness_tracker.finish_turn()
    event_counts: dict[str, int] = {}
    for event in turn_report.events:
        key = event.event_type.value
        event_counts[key] = event_counts.get(key, 0) + 1

    active_session.runtime_session.update_smoothness_state(
        turn_score=turn_report.score,
        labour_index=turn_report.labour_index,
        tok_mode=turn_report.mode,
        event_counts=event_counts,
    )

    active_session.capture_event(
        {
            "event": "response",
            "model": resp_json["model"],
            "request_policy": request_policy,
            "tool_compatible": request_tool_compatible,
            "baseline_only": active_session.runtime_session._baseline_only,
            "persistence_failures": active_session.runtime_session._persistence_failures,
            "fallback_count": int(active_session.tracker.behavior_signals().get("tok_fallback_activated", 0)),
            "behavior_signals": response_signals or {},
            "session_quality": str((active_session.tracker.session_summary() or {}).get("session_quality", "clean")),
            "session_tokens_saved": int((active_session.tracker.session_summary() or {}).get("tokens_saved", 0)),
            "session_savings_pct": float((active_session.tracker.session_summary() or {}).get("savings_pct", 0.0)),
            "last_degradation_reason": str(
                (active_session.tracker.session_summary() or {}).get("last_degradation_reason", "")
            ),
        }
    )
    emit_live_trace(
        active_session,
        "response_processed",
        trace_class="response",
        action="summary_reference" if total_output_saved else "pass_through",
        result="ok",
        expectation="accept_non_exact_reference" if total_output_saved else "accept_pass_through",
        reason=(
            "live metadata-only trace; response artifacts are not captured"
            if total_output_saved
            else "response passed through without live artifact capture"
        ),
        direction="response",
        metadata={
            "output_saved_tokens": int(total_output_saved),
            "behavior_signals": response_signals or {},
            "request_policy": request_policy,
            "tool_compatible": bool(request_tool_compatible),
            "baseline_only": bool(active_session.runtime_session._baseline_only),
        },
    )


def _handle_nonstreaming_failopen(
    exc: Exception,
    active_session: BridgeSession,
    behavior_signals: dict[str, int],
    request_state: dict[str, bool],
    resp_json: dict[str, Any],
    saved_toks: int,
    compressed: bool,
    tool_breakdown: dict[str, int],
    prompt_metrics: dict[str, int],
) -> None:
    if not active_session.fail_open:
        raise exc

    logger.warning("Non-streaming processing error (fail-open): %s", exc)
    behavior_signals["processing_error"] = behavior_signals.get("processing_error", 0) + 1
    behavior_signals["tok_fallback_activated"] = behavior_signals.get("tok_fallback_activated", 0) + 1
    _record_fallback_once(active_session, request_state)
    try:
        model = resp_json.get("model", "")
        usage = resp_json.get("usage", {})
        session_signals = active_session.runtime_session.consume_behavior_signals()
        error_signals: dict[str, int] = {"processing_error": 1, "tok_fallback_activated": 1}
        if session_signals:
            for k, v in session_signals.items():
                error_signals[k] = error_signals.get(k, 0) + v
        if model and usage:
            active_session.tracker.record_call(
                model=model,
                actual_input=usage.get("input_tokens", 0),
                actual_output=usage.get("output_tokens", 0),
                cache_read=usage.get("cache_read_input_tokens", 0),
                cache_write=usage.get("cache_creation_input_tokens", 0),
                input_saved=saved_toks if compressed else 0,
                output_saved=0,
                type_breakdown=tool_breakdown if compressed else None,
                behavior_signals=error_signals,
                prompt_metrics=prompt_metrics if compressed else None,
            )
    except Exception as _exc:
        logger.debug("Failed to record usage in fail-open path: %s", _exc)


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
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Tok gateway is temporarily rate limited; retry later.",
                },
            }
        ),
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": retry_after_header},
    )


async def _build_upstream_rate_limit_response(response: httpx.Response) -> Response:
    content = await response.aread()
    return Response(
        content=content,
        status_code=response.status_code,
        headers=_safe_headers(response.headers),
        media_type=response.headers.get("content-type", "application/json"),
    )


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
    message = {
        "id": resp_json.get("id", ""),
        "type": resp_json.get("type", "message"),
        "role": resp_json.get("role", "assistant"),
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
    }
    message_start = {
        "type": "message_start",
        "message": message,
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
    async def health(request: Request) -> dict[str, Any]:
        explicit_session_key = ""
        if any(
            request.headers.get(header, "").strip()
            for header in ("x-tok-session-id", "x-claude-session-id", "x-codex-session-id", "x-client-session-id")
        ):
            explicit_session_key = session.activate_session_for_request(dict(request.headers), None)
        report_session = (
            session.bound_session_for_key(explicit_session_key) if explicit_session_key else session.reporting_session()
        )
        session_summary = (
            report_session.tracker.session_summary() if explicit_session_key else session.aggregate_session_summary()
        ) or {}
        if explicit_session_key:
            signals = dict(report_session.tracker.behavior_signals())
            for (
                key,
                value,
            ) in report_session.runtime_session.pending_behavior_signals.items():
                signals[key] = signals.get(key, 0) + int(value)
        else:
            signals = session.aggregate_behavior_signals()
        snap = DiagnosticsSnapshot.from_session(
            port=session.port,
            api_base=session.api_base,
            request_policy_default=session.request_policy_default,
            mode_label=_request_policy_mode_label(session.request_policy_default),
            baseline_only=report_session.runtime_session._baseline_only,
            persistence_failures=report_session.runtime_session._persistence_failures,
            session_summary=session_summary,
            signals=signals,
        )
        # Override fields that require the live runtime_session object.
        rs = report_session.runtime_session
        snap = dataclasses.replace(
            snap,
            smoothness_score=rs.latest_turn_smoothness_score,
            labour_index=rs.latest_turn_labour_index,
            current_mode=rs.current_tok_mode.name,
            task_score=rs.current_task_smoothness_score,
            stream_instability_events=sum(v for k, v in rs.smoothness_event_counts.items() if "stream" in k.lower()),
            thinking_mutation_events=int(rs.smoothness_event_counts.get("thinking_block_mutation", 0)),
            stream_recovery_attempt_count=max(
                snap.stream_recovery_attempt_count,
                sum(v for k, v in rs.smoothness_event_counts.items() if "stream_recovery" in k.lower()),
            ),
        )
        health_response = snap.to_health_response()
        health_response["session_count"] = 1 if explicit_session_key else session.aggregate_session_count()
        health_response["capability"] = asdict(
            build_capability_manifest(bridge_mode=str(health_response.get("mode", "unknown")))
        )
        return health_response

    @app.post("/reset-session")
    async def reset_session_endpoint(request: Request) -> dict[str, str]:
        """Reset the caller's session bucket, or all buckets with scope=all."""
        if request.query_params.get("scope", "").strip().lower() == "all":
            session.reset_all_sessions()
            return {"status": "ok", "action": "session_reset_all"}
        body_bytes = await request.body()
        request_body_obj: dict[str, Any] | None = None
        if body_bytes:
            with contextlib.suppress(Exception):
                decoded_body = json.loads(body_bytes)
                if isinstance(decoded_body, dict):
                    request_body_obj = decoded_body
        # Always resolve the caller's bucket the same way as normal requests
        # (header-based keys or auto keys derived from auth/user-agent/message seed).
        session.activate_session_for_request(dict(request.headers), request_body_obj)
        session.reset_active_session()
        return {"status": "ok", "action": "session_reset"}

    @app.post("/flush-ledger")
    async def flush_ledger_endpoint() -> dict[str, str]:
        """Persist live session buckets into the lifetime ledger before shutdown."""
        session.merge_all_trackers_to_ledger()
        return {"status": "ok", "action": "ledger_flushed"}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def bridge(request: Request, path: str) -> Response:
        body_bytes = await request.body()
        original_body_bytes = body_bytes
        request_state = {"fallback_recorded": False}
        active_session = session
        request_body_obj: dict[str, Any] | None = None
        if path.startswith("v1/") and request.method == "POST" and body_bytes:
            with contextlib.suppress(Exception):
                decoded_body = json.loads(body_bytes)
                if isinstance(decoded_body, dict):
                    request_body_obj = decoded_body
        if path.startswith("v1/"):
            session_key = session.activate_session_for_request(dict(request.headers), request_body_obj)
            active_session = session.bound_session_for_key(session_key)

        if path.startswith("v1/") and _is_rate_limited(session):
            logger.warning(
                "rate_limit_throttle_active: blocking request to %s for %.2fs",
                path,
                _rate_limit_throttle_remaining(session),
            )
            return _build_rate_limit_response(_rate_limit_throttle_remaining(session))

        active_session.smoothness_tracker.start_turn(
            task_id=f"session_{active_session.runtime_session.bridge_memory.turn}"
        )

        skip = {
            "host",
            "content-length",
            "accept-encoding",
            "connection",
            "x-tok-tool-compatible",
            "x-tok-session-id",
            "x-claude-session-id",
            "x-codex-session-id",
            "x-client-session-id",
        }
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        headers["host"] = _upstream_host(active_session.api_base)

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
                    session=active_session,
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

                    if active_session.runtime_session.current_tok_mode == TokMode.LOSSLESS_TASK_MODE:
                        logger.warning("LOSSLESS_TASK_MODE active: preserving full task flow for emergency mode")

                    _note_request_policy_recovery_watch(active_session, behavior_signals)
                    if behavior_signals.get("smoothness_history_winnowing_active_loop"):
                        with contextlib.suppress(Exception):
                            active_session.smoothness_tracker.record(
                                SmoothnessEventType.HISTORY_WINNOWING_ACTIVE_LOOP,
                            )
                    if behavior_signals.get("smoothness_prompt_optimization_active_task"):
                        with contextlib.suppress(Exception):
                            active_session.smoothness_tracker.record(
                                SmoothnessEventType.PROMPT_OPTIMIZATION_ACTIVE_TASK,
                            )
                    if behavior_signals.get("repeat_file_read", 0) > 0:
                        with contextlib.suppress(Exception):
                            active_session.smoothness_tracker.record(
                                SmoothnessEventType.REPEATED_ACTIVE_FILE_READ,
                                {"count": behavior_signals.get("repeat_file_read", 0)},
                            )

                    active_session.capture_request(
                        {
                            "event": "request",
                            "messages": messages,
                            "system": body.get("system", ""),
                            "model": request_model,
                            "tool_compatible": request_tool_compatible,
                            "request_policy": request_policy,
                            "plan_finalization": bool(behavior_signals.get("plan_finalization_turn", 0)),
                            "plan_finalization_passthrough": bool(
                                behavior_signals.get("plan_finalization_passthrough", 0)
                            ),
                            "prompt_metrics": prompt_metrics,
                            "original_body_bytes": len(provider_safe_original_body_bytes),
                            "prepared_body_bytes": len(json.dumps(body).encode()),
                        }
                    )
                    if not behavior_signals.get("plan_finalization_passthrough", 0):
                        body = apply_anthropic_optimizations(body, behavior_signals=behavior_signals)
                    body_bytes = json.dumps(body).encode()
                    emit_live_trace(
                        active_session,
                        "request_prepared",
                        trace_class="message",
                        action="summary_reference" if compressed else "pass_through",
                        result="ok",
                        expectation="accept_non_exact_reference" if compressed else "accept_pass_through",
                        reason=(
                            "live metadata-only trace; request artifacts are not captured"
                            if compressed
                            else "request passed through without live artifact capture"
                        ),
                        direction="request",
                        metadata={
                            "compressed": bool(compressed),
                            "input_saved_tokens": int(saved_toks if compressed else 0),
                            "request_policy": request_policy,
                            "mode": _request_policy_mode_label(request_policy),
                            "tool_compatible": bool(request_tool_compatible),
                            "behavior_signals": behavior_signals,
                            "prompt_metrics": prompt_metrics,
                            "prepared_body_bytes": len(body_bytes),
                        },
                    )

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
                        active_session.runtime_session.reset_fallback_count()
                else:
                    body = provider_safe_original_body
                    body_bytes = provider_safe_original_body_bytes
                    active_session.capture_request(
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
                if active_session.fail_open:
                    logger.warning(
                        "tok_fallback_activated: JSON decode error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["json_decode_error"] = 1
                    _record_fallback_once(active_session, request_state)
                else:
                    raise
            except (KeyError, AttributeError) as exc:
                if active_session.fail_open:
                    logger.warning(
                        "tok_fallback_activated: data structure error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["data_structure_error"] = 1
                    _record_fallback_once(active_session, request_state)
                else:
                    raise
            except (MemoryError, OverflowError) as exc:
                if active_session.fail_open:
                    logger.error(
                        "tok_fallback_activated: critical system error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["critical_system_error"] = 1
                    _record_fallback_once(active_session, request_state)
                else:
                    raise
            except Exception as exc:
                if active_session.fail_open:
                    logger.error(
                        "tok_fallback_activated: unexpected error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    behavior_signals["unexpected_error"] = 1
                    _record_fallback_once(active_session, request_state)
                else:
                    logger.error("Unexpected error in request processing: %s", exc)
                    raise

        target_url = _upstream_target_url(active_session.api_base, path, request.url.query)

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

                current_mode = active_session.runtime_session.current_tok_mode
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
                    active_session,
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
                _note_request_policy_recovery_watch(active_session, retry_signals)
                if response.status_code == 429:
                    _record_rate_limit_hit(session)
                    try:
                        # aread() drains and closes the streaming response; only client needs cleanup.
                        rate_limit_response = await _build_upstream_rate_limit_response(response)
                    finally:
                        await _close_streaming_setup_resources(None, client)
                    return rate_limit_response
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
                    _record_fallback_once(active_session, request_state)
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
                    if behavior_signals.get("plan_finalization_passthrough", 0):
                        return StreamingResponse(
                            passthrough_stream_impl(
                                active_session,
                                client,
                                response,
                                input_saved_tokens=0,
                                behavior_signals=behavior_signals or None,
                                prompt_metrics=None,
                                client_owned=True,
                            ),
                            status_code=response.status_code,
                            headers=resp_headers,
                            media_type=response.headers.get("content-type", "text/event-stream"),
                        )
                    return StreamingResponse(
                        buffer_strip_restream_impl(
                            active_session,
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
                if active_session.fail_open:
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
                            active_session,
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
                        _note_request_policy_recovery_watch(active_session, retry_signals)
                        if _retried:
                            _record_fallback_once(active_session, request_state)
                        if path == "v1/messages":
                            # Ownership transfers here. The streaming implementation must close both response and client.
                            return StreamingResponse(
                                buffer_strip_restream_impl(
                                    active_session,
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
                active_session,
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
            _note_request_policy_recovery_watch(active_session, retry_signals)
            if response.status_code == 429:
                _record_rate_limit_hit(session)
                return await _build_upstream_rate_limit_response(response)
            if retried_without_tok:
                compressed, saved_toks, tool_breakdown, prompt_metrics = _handle_retried_without_tok(
                    behavior_signals,
                    active_session,
                    request_state,
                )

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
                    full_response_text = ""

                    logger.info(
                        "Raw response content: %s",
                        resp_json.get("content", [])[:3] if resp_json.get("content") else [],
                    )

                    for block in resp_json.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content = block.get("text")
                            if isinstance(text_content, str):
                                full_response_text += text_content

                    response_signals, total_output_saved = _build_response_signals(
                        full_response_text,
                        resp_json,
                        active_session,
                        behavior_signals,
                        request_tool_compatible,
                    )
                    _rebuild_and_record_response(
                        resp_json,
                        active_session,
                        saved_toks,
                        compressed,
                        tool_breakdown,
                        response_signals,
                        prompt_metrics,
                        total_output_saved,
                        request_policy,
                        request_tool_compatible,
                    )
                    content = json.dumps(resp_json).encode()
                except Exception as exc:
                    _handle_nonstreaming_failopen(
                        exc,
                        active_session,
                        behavior_signals,
                        request_state,
                        resp_json,
                        saved_toks,
                        compressed,
                        tool_breakdown,
                        prompt_metrics,
                    )

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
