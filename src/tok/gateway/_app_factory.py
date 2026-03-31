from __future__ import annotations

"""Gateway app-construction helpers behind the public interface."""

import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from ..runtime.pipeline.request_validation import (
    canonicalize_anthropic_bridge_body,
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
) -> AsyncIterator[bytes]:
    """Buffer the full SSE stream, translate Tok -> readable English/tool_use, re-emit."""
    try:
        raw = b""
        async for chunk in response.aiter_bytes():
            raw += chunk

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

        tool_blocks = _materialize_stream_tool_blocks(
            stream_blocks, stream_order
        )
        content_blocks = _response_contract_for_mode(
            full_text, tool_compatible=tool_compatible
        ).content_blocks
        logger.info(
            "Translated %d content blocks from %d chars",
            len(content_blocks) + len(tool_blocks),
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
                behavior_signals=behavior_signals or None,
                tool_compatible=tool_compatible,
            )
            content_blocks = processed.content_blocks + tool_blocks
            response_signals = processed.behavior_signals

            logger.info("Response mode: %s", processed.mode)
            logger.info("Response signals: %s", response_signals)
            logger.info("Content blocks count: %d", len(content_blocks))
        if sse_model != "unknown" and sse_usage:
            if not full_text:
                processed = _RUNTIME.process_response(
                    "",
                    model=sse_model,
                    session=session.runtime_session,
                    behavior_signals=behavior_signals or None,
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
        await client.aclose()


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
        stream: bool = False,
        compressed_request: bool = False,
    ) -> tuple[httpx.Response, bool]:
        request_obj = client.build_request(
            method, url, headers=headers, content=content
        )
        response = await client.send(request_obj, stream=stream)
        retried_without_tok = False

        logger.warning(
            "Fail-open check: status=%d, compressed=%s, has_orig=%s, fail_open=%s",
            response.status_code,
            compressed_request,
            original_content is not None,
            session.fail_open,
        )

        if (
            response.status_code == 400
            and compressed_request
            and original_content is not None
            and _payloads_materially_differ(content, original_content)
            and session.fail_open
        ):
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
            logger.warning(
                "Upstream 400 after Tok request preparation: %s; retrying with original payload",
                error_text[:500],
            )
            await response.aclose()
            request_obj = client.build_request(
                method, url, headers=headers, content=original_content
            )
            response = await client.send(request_obj, stream=stream)
            retried_without_tok = True
        return response, retried_without_tok

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

        if path == "v1/messages" and request.method == "POST" and body_bytes:
            try:
                body = json.loads(body_bytes)
                original_body = (
                    copy.deepcopy(body) if isinstance(body, dict) else body
                )
                messages = body.get("messages", [])
                request_model = str(body.get("model", ""))
                tok_tool_header = request.headers.get(
                    "x-tok-tool-compatible", ""
                )
                if tok_tool_header.lower() in {"0", "false", "off", "no"}:
                    request_tool_compatible = False
                elif session.runtime_session._baseline_only:
                    request_tool_compatible = False
                    behavior_signals["baseline_only_session"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    logger.warning(
                        "tok_fallback_activated: session is in baseline-only mode, serving without compression"
                    )
                else:
                    request_tool_compatible = session.tool_compatible_default

                logger.info(
                    "Request mode: model=%s, tool_compatible=%s (tools present: %s, header=%s)",
                    request_model,
                    request_tool_compatible,
                    bool(body.get("tools")),
                    tok_tool_header or "<unset>",
                )

                prepared = _RUNTIME.prepare_request(
                    RuntimeRequest(
                        model=request_model,
                        messages=messages,
                        system=body.get("system", ""),
                        adapter_kind="claude-bridge",
                        tool_compatible=request_tool_compatible,
                    ),
                    session.runtime_session,
                    result_cache=session.result_cache,
                )
                compressed = prepared.compressed
                saved_toks = prepared.input_saved_tokens
                tool_breakdown = prepared.type_breakdown
                behavior_signals = prepared.behavior_signals
                prompt_metrics = {
                    "baseline_prompt_tokens": prepared.baseline_prompt_tokens,
                    "prepared_prompt_tokens": prepared.prepared_prompt_tokens,
                    "saved_prompt_tokens": prepared.saved_prompt_tokens,
                    "hot_hint_tokens_added": prepared.hot_hint_tokens_added,
                    "reacquisition_tokens_avoided_estimate": prepared.reacquisition_tokens_avoided_estimate,
                }
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
                    canonical_body,
                    bridge_canonicalized,
                    bridge_signals,
                ) = canonicalize_anthropic_bridge_body(body)
                strict_failures = validate_anthropic_bridge_body(
                    canonical_body
                )
                request_fingerprint = _request_fingerprint_diff(
                    headers, canonical_body, original_body
                )
                should_log_preflight = bool(
                    compressed or bridge_canonicalized or strict_failures
                )
                if bridge_signals:
                    behavior_signals.update(
                        {
                            key: behavior_signals.get(key, 0) + value
                            for key, value in bridge_signals.items()
                        }
                    )
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
                if strict_failures:
                    _log_bridge_body_structure(
                        "bridge_preflight_rejected",
                        body=canonical_body,
                        headers=headers,
                        original_body=original_body,
                        compressed_request=compressed,
                        canonicalized_changed=bridge_canonicalized,
                        strict_failures=strict_failures,
                        reverted_to_original=True,
                    )
                    logger.warning(
                        "tok_bridge_preflight_rejected: reverting rewritten bridge body to original request"
                    )
                    body = copy.deepcopy(original_body)
                    behavior_signals["tok_bridge_preflight_rejected"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    compressed = False
                    saved_toks = 0
                    tool_breakdown = {}
                    _record_fallback_once(session, request_state)
                else:
                    body = canonical_body
                    if should_log_preflight:
                        _log_bridge_body_structure(
                            "bridge_preflight_ready",
                            body=body,
                            headers=headers,
                            original_body=original_body,
                            compressed_request=compressed,
                            canonicalized_changed=bridge_canonicalized,
                            strict_failures=[],
                            reverted_to_original=False,
                        )
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

            except Exception as exc:
                if session.fail_open:
                    logger.warning(
                        "tok_fallback_activated: request processing error, serving without compression: %s",
                        exc,
                    )
                    behavior_signals["processing_error"] = 1
                    behavior_signals["tok_fallback_activated"] = 1
                    _record_fallback_once(session, request_state)
                else:
                    raise

        target_url = f"{ANTHROPIC_API_BASE}/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        is_streaming = False
        try:
            body_dict = json.loads(body_bytes)
            is_streaming = bool(body_dict.get("stream", False))
        except Exception:
            pass

        if is_streaming:
            client = httpx.AsyncClient(timeout=300.0)
            try:
                (
                    response,
                    retried_without_tok,
                ) = await _send_with_tok_fail_open_retry(
                    client,
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body_bytes,
                    original_content=original_body_bytes,
                    stream=True,
                    compressed_request=compressed,
                )
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
                    logger.warning(
                        "tok_fallback_activated: upstream 400 retry, serving without compression"
                    )
                    _record_fallback_once(session, request_state)
                resp_headers = _safe_headers(response.headers)

                if path == "v1/messages":
                    return StreamingResponse(
                        buffer_strip_restream_impl(
                            session,
                            client,
                            response,
                            input_saved_tokens=saved_toks if compressed else 0,
                            type_breakdown=tool_breakdown
                            if compressed
                            else None,
                            behavior_signals=behavior_signals or None,
                            prompt_metrics=prompt_metrics
                            if compressed
                            else None,
                            tool_compatible=request_tool_compatible,
                        ),
                        status_code=response.status_code,
                        headers=resp_headers,
                        media_type=response.headers.get(
                            "content-type", "text/event-stream"
                        ),
                    )
            except Exception as e:
                logger.error(
                    "Streaming error in Tok bridge: %s", str(e), exc_info=True
                )
                await client.aclose()

        async with httpx.AsyncClient(timeout=300.0) as client:
            (
                response,
                retried_without_tok,
            ) = await _send_with_tok_fail_open_retry(
                client,
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes,
                original_content=original_body_bytes,
                compressed_request=compressed,
            )
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
                        if block.get("type") != "text"
                    ]

                    logger.info(
                        "Raw response content: %s",
                        resp_json.get("content", [])[:3],
                    )

                    for block in resp_json.get("content", []):
                        if block.get("type") == "text":
                            full_response_text += block["text"]

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
