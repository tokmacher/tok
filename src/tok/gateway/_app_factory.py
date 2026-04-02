from __future__ import annotations

"""Gateway app-construction helpers behind the public interface."""

import asyncio  # noqa: F401
import copy
import json
import random  # noqa: F401
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from ..runtime.pipeline.request_validation import normalize_tool_use_blocks
from ..universal_runtime import RuntimeRequest
from . import (
    ANTHROPIC_API_BASE,
    BridgeSession,
    _RUNTIME,
    _record_fallback_once,
    _request_policy_mode_label,
    logger,
)
from ._bridge_comparison import _safe_headers
from ._bridge_preflight import (
    _local_rate_limit_response,
    _run_bridge_preflight,
)
from ._bridge_request_handler import send_with_tok_fail_open_retry
from ._bridge_streaming import buffer_strip_restream_impl

__all__ = ["buffer_strip_restream_impl", "create_app_impl"]


def _note_request_policy_recovery_watch(
    session: BridgeSession, signals: dict[str, int] | None
) -> None:
    if not signals:
        return
    if signals.get("stream_recovery_retry", 0) or signals.get(
        "stream_recovery_fallback", 0
    ):
        session.runtime_session.note_request_policy_stream_recovery()
    if (
        signals.get("fail_open_retry_upstream_pairing_disagreement", 0)
        or signals.get("tok_bridge_provider_pairing_risk_detected", 0)
        or signals.get("tok_bridge_pairing_degraded_to_provider_safe", 0)
        or signals.get(
            "tok_bridge_assistant_tool_use_text_interleaving_blocked", 0
        )
        or signals.get("tok_bridge_invalid_tool_history_recovery", 0)
        or signals.get("tok_bridge_invalid_tool_history_quarantined", 0)
        or signals.get("tok_bridge_invalid_tool_history_blocked", 0)
        or signals.get("tok_history_pairing_safety_degraded", 0)
    ):
        session.runtime_session.note_request_policy_tool_mode_recovery()


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
            "mode": _request_policy_mode_label(session.request_policy_default),
            "request_policy": session.request_policy_default,
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
            "tool_history_repaired_count": int(
                session_summary.get("tool_history_repaired_count", 0)
            ),
            "tool_history_pairing_repaired_count": int(
                session_summary.get("tool_history_pairing_repaired_count", 0)
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
            "provider_pairing_disagreement_count": int(
                session_summary.get("provider_pairing_disagreement_count", 0)
            ),
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
            "request_policy_natural_first_count": int(
                session_summary.get("request_policy_natural_first_count", 0)
            ),
            "request_policy_tool_compatible_count": int(
                session_summary.get("request_policy_tool_compatible_count", 0)
            ),
            "request_policy_escalations_count": int(
                session_summary.get("request_policy_escalations_count", 0)
            ),
            "request_policy_deescalations_count": int(
                session_summary.get("request_policy_deescalations_count", 0)
            ),
            "request_policy_interleaving_downgrades_count": int(
                session_summary.get(
                    "request_policy_interleaving_downgrades_count", 0
                )
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
                    signals.get(
                        "request_policy_reason_structured_tool_loop", 0
                    ),
                )
            ),
            "request_policy_held_by_recovery_count": int(
                session_summary.get(
                    "request_policy_held_by_recovery_count",
                    signals.get("request_policy_held_by_recovery", 0)
                    + signals.get(
                        "request_policy_recovery_sticky_continuations", 0
                    ),
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
        request_policy = "forced_baseline"
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

                (
                    provider_safe_original_body,
                    behavior_signals,
                    source_retry_forbidden,
                    preflight_response,
                ) = _run_bridge_preflight(
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

                request_model = str(
                    provider_safe_original_body.get("model", "")
                )
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
                        request_policy = (
                            "forced_baseline"
                            if session.request_policy_default
                            == "forced_baseline"
                            else "natural_first"
                        )
                    elif session.runtime_session._baseline_only:
                        request_tool_compatible = False
                        request_policy = "forced_baseline"
                        behavior_signals["baseline_only_session"] = 1
                        behavior_signals["tok_fallback_activated"] = 1
                        logger.warning(
                            "tok_fallback_activated: session is in baseline-only mode, serving without compression"
                        )
                    else:
                        request_tool_compatible = True
                        request_policy = session.request_policy_default

                    logger.info(
                        "Request mode: model=%s, request_policy=%s, tool_compatible_allowed=%s (tools present: %s, header=%s)",
                        request_model,
                        request_policy,
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
                            request_policy=request_policy,
                            request_has_tools=bool(
                                provider_safe_original_body.get("tools")
                            ),
                        ),
                        session.runtime_session,
                        result_cache=session.result_cache,
                    )
                    request_policy = prepared.request_policy
                    request_tool_compatible = (
                        prepared.effective_tool_compatible
                    )
                    compressed = prepared.compressed
                    saved_toks = prepared.input_saved_tokens
                    tool_breakdown = prepared.type_breakdown
                    behavior_signals = dict(prepared.behavior_signals)
                    _note_request_policy_recovery_watch(
                        session, behavior_signals
                    )
                    for key, value in source_behavior_signals.items():
                        behavior_signals[key] = (
                            behavior_signals.get(key, 0) + value
                        )
                    _note_request_policy_recovery_watch(
                        session, behavior_signals
                    )
                    prompt_metrics = {
                        "baseline_prompt_tokens": prepared.baseline_prompt_tokens,
                        "prepared_prompt_tokens": prepared.prepared_prompt_tokens,
                        "saved_prompt_tokens": prepared.saved_prompt_tokens,
                        "hot_hint_tokens_added": prepared.hot_hint_tokens_added,
                        "reacquisition_tokens_avoided_estimate": prepared.reacquisition_tokens_avoided_estimate,
                    }
                    logger.info(
                        "Prepared request policy: request_policy=%s, effective_tool_compatible=%s, escalated=%s",
                        request_policy,
                        request_tool_compatible,
                        prepared.request_policy_escalated,
                    )
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
                            "request_policy": request_policy,
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
                logger.error(
                    "tok_fallback_activated: critical system error, serving without compression: %s",
                    exc,
                )
                behavior_signals["processing_error"] = 1
                behavior_signals["tok_fallback_activated"] = 1
                behavior_signals["critical_system_error"] = 1
                _record_fallback_once(session, request_state)
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
                    logger.error(
                        "Unexpected error in request processing: %s", exc
                    )
                    raise

        target_url = f"{ANTHROPIC_API_BASE}/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        if session.is_rate_limited_locally():
            retry_after_seconds = (
                session.local_rate_limit_retry_after_seconds()
            )
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
            logger.debug("Failed to parse JSON for streaming detection")
        except Exception as exc:
            logger.debug("Unexpected error during JSON parsing: %s", exc)

        if is_streaming:
            async with httpx.AsyncClient(timeout=300.0) as client:
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
                    )
                    for key, value in retry_signals.items():
                        behavior_signals[key] = (
                            behavior_signals.get(key, 0) + value
                        )
                    _note_request_policy_recovery_watch(session, retry_signals)
                    if session.is_rate_limited_locally():
                        behavior_signals[
                            "rate_limit_local_throttle_opened"
                        ] = 1
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
                        for key, value in retry_signals.items():
                            behavior_signals[key] = (
                                behavior_signals.get(key, 0) + value
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
                                behavior_signals[key] = (
                                    behavior_signals.get(key, 0) + value
                                )
                            _note_request_policy_recovery_watch(
                                session, retry_signals
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
            )
            for key, value in retry_signals.items():
                behavior_signals[key] = behavior_signals.get(key, 0) + value
            _note_request_policy_recovery_watch(session, retry_signals)
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
                for key, value in retry_signals.items():
                    behavior_signals[key] = (
                        behavior_signals.get(key, 0) + value
                    )
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
                        if isinstance(block, dict)
                        and block.get("type") != "text"
                    ]
                    passthrough_blocks, passthrough_signals = (
                        normalize_tool_use_blocks(
                            passthrough_blocks, seed_prefix="toolu_upstream"
                        )
                    )
                    for key, value in passthrough_signals.items():
                        behavior_signals[key] = (
                            behavior_signals.get(key, 0) + value
                        )

                    logger.info(
                        "Raw response content: %s",
                        resp_json.get("content", [])[:3]
                        if resp_json.get("content")
                        else [],
                    )

                    for block in resp_json.get("content", []):
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
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
                    else:
                        session_signals = (
                            session.runtime_session.consume_behavior_signals()
                        )
                        if session_signals:
                            response_signals = dict(session_signals)

                    _note_request_policy_recovery_watch(
                        session, response_signals
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
                                "request_policy": request_policy,
                                "tool_compatible": request_tool_compatible,
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
