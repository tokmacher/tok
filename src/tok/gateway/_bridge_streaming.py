from __future__ import annotations

"""Streaming response helpers for the Tok gateway."""

import hashlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpcore
import httpx

from ..runtime.pipeline.request_validation import normalize_tool_use_blocks
from ..runtime.policy.translator import IS_TOK
from . import (
    BridgeSession,
    _RUNTIME,
    _materialize_stream_tool_blocks,
    _record_fallback_once,
    _response_contract_for_mode,
    logger,
)

__all__ = ["buffer_strip_restream_impl"]

_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT: int = int(
    os.getenv("TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT", "2")
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

        processed: Any | None = (
            None  # Will hold ProcessedRuntimeResponse when available
        )

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
            if read_error is not None:
                stream_behavior_signals["stream_recovery_read_error"] = 1
            else:
                stream_behavior_signals["stream_recovery_empty_success"] = 1
            session.runtime_session._stream_recovery_reacquisition_budget = 1
            session.runtime_session._stream_recovery_history_floor_budget = 1
            session.runtime_session.note_request_policy_stream_recovery()
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
                            # Initialize retry_response_signals to accumulate across branches
                            retry_response_signals: dict[str, int] = dict(
                                stream_behavior_signals
                            )
                            if retry_text:
                                retry_processed = _RUNTIME.process_response(
                                    retry_text,
                                    model=str(retry_json.get("model", "")),
                                    session=session.runtime_session,
                                    behavior_signals=stream_behavior_signals
                                    or None,
                                    tool_compatible=tool_compatible,
                                )
                                _merge_signal_counts(
                                    retry_response_signals,
                                    retry_processed.behavior_signals,
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
                                _merge_signal_counts(
                                    retry_response_signals,
                                    retry_processed.behavior_signals,
                                )
                                translated_blocks = passthrough_blocks
                            else:
                                # Keep response_signals as initialized from stream_behavior_signals
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
                                _merge_signal_counts(
                                    response_signals,
                                    recovery_success_signals,
                                )
                                retry_usage = retry_json.get("usage", {})
                                retry_model = str(retry_json.get("model", ""))
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
                # processed was assigned in the outer full_text block at lines 203-209
                # Ensure we have a valid processed object before accessing attributes
                if processed is not None:
                    output_saved = processed.output_saved_tokens
                    response_signals = processed.behavior_signals
                else:
                    output_saved = 0
                    response_signals = stream_behavior_signals or {}

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
