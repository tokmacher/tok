"""Streaming response helpers for the Tok gateway."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import TYPE_CHECKING, Any

import httpcore
import httpx

from tok.runtime.pipeline.request_validation import normalize_tool_use_blocks
from tok.runtime.policy.translator import IS_TOK
from tok.runtime.smoothness.models import SmoothnessEventType

from . import (
    _RUNTIME,
    BridgeSession,
    _materialize_stream_tool_blocks,
    _record_fallback_once,
    _response_contract_for_mode,
    logger,
)
from ._signal_constants import _merge_signal_counts

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

__all__ = ["buffer_strip_restream_impl", "passthrough_stream_impl"]


def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer config %s=%r; using fallback %d", name, raw, fallback)
        return fallback


_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT: int = _env_int("TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT", 2)


def _tool_use_only_signature(blocks: list[dict[str, Any]]) -> str:
    """
    Create a signature from tool_use blocks for loop detection.

    Returns a hashable string signature based on tool names, input keys,
    and input value hashes to detect repeated identical tool_use-only
    recovery patterns.
    """
    tool_uses = [block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"]
    if not tool_uses:
        return ""
    parts: list[str] = []
    for tool in tool_uses:
        name = str(tool.get("name", ""))
        input_dict = tool.get("input", {})
        if isinstance(input_dict, dict):
            keys = ",".join(sorted(input_dict.keys()))
            try:
                value_hash = hashlib.sha256(json.dumps(input_dict, sort_keys=True).encode()).hexdigest()[:8]
            except Exception:
                value_hash = ""
        else:
            keys = ""
            value_hash = ""
        parts.append(f"{name}:{keys}:{value_hash}")
    return "|".join(parts)


def _emit_sse_block(i: int, block: dict[str, Any]) -> list[bytes]:
    """Emit SSE events for a single content block (text, tool_use, or thinking)."""
    events: list[bytes] = []
    block_type = block.get("type", "text")

    if block_type == "thinking":
        thinking_text = block.get("thinking", "")
        signature = block.get("signature", "")
        start = {
            "type": "content_block_start",
            "index": i,
            "content_block": {
                "type": "thinking",
                "thinking": "",
            },
        }
        events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode())
        if thinking_text:
            delta = {
                "type": "content_block_delta",
                "index": i,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": thinking_text,
                },
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
        if signature:
            sig_delta = {
                "type": "content_block_delta",
                "index": i,
                "delta": {
                    "type": "signature_delta",
                    "signature": signature,
                },
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(sig_delta)}\n\n".encode())
        stop = {"type": "content_block_stop", "index": i}
        events.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode())

    elif block_type == "redacted_thinking":
        start = {
            "type": "content_block_start",
            "index": i,
            "content_block": {
                "type": "redacted_thinking",
                "data": block.get("data"),
            },
        }
        events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode())
        stop = {"type": "content_block_stop", "index": i}
        events.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode())

    elif block_type == "text":
        start = {
            "type": "content_block_start",
            "index": i,
            "content_block": {"type": "text", "text": ""},
        }
        events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode())
        delta = {
            "type": "content_block_delta",
            "index": i,
            "delta": {"type": "text_delta", "text": block.get("text", "")},
        }
        events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
        stop = {"type": "content_block_stop", "index": i}
        events.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode())

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
        events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode())
        if isinstance(tool_input, dict):
            delta = {
                "type": "content_block_delta",
                "index": i,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tool_input),
                },
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
        stop = {"type": "content_block_stop", "index": i}
        events.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode())

    return events


def _record_stream_read_error(
    session: BridgeSession,
    stage_label: str,
    read_error_text: str,
) -> None:
    """Record stream read error with shared bookkeeping and observability."""
    logger.warning(
        "Stream read error during %s: %s",
        stage_label,
        read_error_text,
    )
    with contextlib.suppress(Exception):
        session.smoothness_tracker.record(
            SmoothnessEventType.STREAM_READ_ERROR,
            {"error": read_error_text},
        )

    runtime_session = session.runtime_session
    if not hasattr(runtime_session, "_stream_read_error_consecutive_count"):
        runtime_session._stream_read_error_consecutive_count = 0
    if not hasattr(runtime_session, "_stream_read_error_last_stage"):
        runtime_session._stream_read_error_last_stage = ""

    runtime_session._stream_read_error_consecutive_count += 1
    runtime_session._stream_read_error_last_stage = stage_label


def _clear_stream_read_error_streak(session: BridgeSession) -> None:
    """Clear shared stream read-error streak state after successful visible completion."""
    runtime_session = session.runtime_session
    runtime_session._stream_read_error_consecutive_count = 0
    runtime_session._stream_read_error_last_stage = ""


def _stream_recovery_allowed_now(
    session: BridgeSession,
) -> tuple[bool, str]:
    """Return whether stream recovery may start now."""
    runtime_session = session.runtime_session
    cooldown_remaining = getattr(
        runtime_session,
        "_stream_recovery_cooldown_remaining",
        0,
    )
    if cooldown_remaining > 0:
        return False, "cooldown_active"
    return True, "eligible"


def _observe_stream_tool_blocks(session: BridgeSession, tool_blocks: list[dict[str, Any]]) -> dict[str, int]:
    """Record semantic runtime side effects for streamed tool-only responses."""
    signals: dict[str, int] = {}
    runtime_session = session.runtime_session
    for block in tool_blocks:
        if block.get("type") != "tool_use" or not block.get("name"):
            continue
        tool_name = str(block["name"])
        runtime_session._tool_names_seen.add(tool_name)
        tool_input = block.get("input", {})
        input_key = next(iter(tool_input.values()), "") if isinstance(tool_input, dict) and tool_input else ""
        if runtime_session.observe_tool_action(tool_name, str(input_key)[:120]):
            signals["loop_detected"] = 1
        if tool_name.lower() in ("edit_file", "edit", "write_file", "write", "replace", "create_file"):
            edited_path = ""
            if isinstance(tool_input, dict):
                edited_path = str(
                    tool_input.get("path") or tool_input.get("file_path") or tool_input.get("filename") or ""
                )
            if edited_path:
                from tok.runtime.repeat_targets import normalize_path_target

                runtime_session.mark_file_edited(normalize_path_target(edited_path))
    return signals


async def passthrough_stream_impl(
    session: BridgeSession,
    client: httpx.AsyncClient,
    response: httpx.Response,
    input_saved_tokens: int = 0,
    behavior_signals: dict[str, int] | None = None,
    prompt_metrics: dict[str, int] | None = None,
    client_owned: bool = False,
) -> AsyncIterator[bytes]:
    """
    Stream-through mode: yield raw SSE chunks without buffering.

    Used when tool_compatible=False (baseline mode) to preserve real-time
    streaming UX for clients like Claude Code that feed bytes incrementally
    to their SSE parsers.
    """
    sse_model: str = "unknown"
    sse_usage: dict[str, Any] = {}
    read_error_occurred: bool = False
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
            chunk_text = chunk.decode("utf-8", errors="replace")
            for line in chunk_text.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    d = json.loads(line[6:])
                    etype = d.get("type", "")
                    if etype == "message_start":
                        msg = d.get("message", {})
                        sse_model = msg.get("model", sse_model)
                        sse_usage = msg.get("usage", sse_usage)
                    elif etype == "message_delta":
                        sse_usage.update(d.get("usage", {}))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug("SSE parse error (usage extraction): %s", exc)
    except (httpx.ReadError, httpcore.ReadError) as e:
        read_error_occurred = True
        read_error = str(e)
        _record_stream_read_error(session, "passthrough", read_error)
    finally:
        response_aclose: Callable[[], Awaitable[None]] | None = getattr(response, "aclose", None)
        if callable(response_aclose):
            await response_aclose()
        if client_owned:
            client_aclose: Callable[[], Awaitable[None]] | None = getattr(client, "aclose", None)
            if callable(client_aclose):
                await client_aclose()
        if not read_error_occurred:
            _clear_stream_read_error_streak(session)
        if sse_model != "unknown" and sse_usage:
            session.tracker.record_call(
                model=sse_model,
                actual_input=sse_usage.get("input_tokens", 0),
                actual_output=sse_usage.get("output_tokens", 0),
                cache_read=sse_usage.get("cache_read_input_tokens", 0),
                cache_write=sse_usage.get("cache_creation_input_tokens", 0),
                input_saved=input_saved_tokens,
                output_saved=0,
                type_breakdown=None,
                behavior_signals=behavior_signals or None,
                prompt_metrics=prompt_metrics or None,
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
    client_owned: bool = False,
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
            _record_stream_read_error(session, "buffering", read_error)
        text = raw.decode("utf-8", errors="replace")
        sse_events = text.split("\n\n")

        accumulated: list[str] = []
        sse_model: str = "unknown"
        sse_usage: dict[str, Any] = {}
        stream_blocks: dict[int, dict[str, Any]] = {}
        stream_order: list[int] = []
        response_signals: dict[str, int] = {}

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
                        if not isinstance(index, int) or not isinstance(block, dict):
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
                        elif block_type == "thinking":
                            stream_blocks[index] = {
                                "type": "thinking",
                                "thinking": str(block.get("thinking", "")),
                                "signature": "",
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
                        block = stream_blocks.setdefault(index, {"type": "text", "text": ""})
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
                        elif delta_type == "thinking_delta":
                            block["type"] = "thinking"
                            block["thinking"] = str(block.get("thinking", "")) + delta.get("thinking", "")
                        elif delta_type == "signature_delta":
                            block["type"] = "thinking"
                            block["signature"] = str(block.get("signature", "")) + delta.get("signature", "")
                        logger.debug(
                            "Delta type: %s, partial: %s",
                            delta_type,
                            str(delta)[:50],
                        )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug("SSE parse error (content accumulation): %s", exc)

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

        processed: Any | None = None  # Will hold ProcessedRuntimeResponse when available

        tool_blocks = _materialize_stream_tool_blocks(stream_blocks, stream_order)
        if tool_blocks:
            _merge_signal_counts(stream_behavior_signals, _observe_stream_tool_blocks(session, tool_blocks))
        thinking_blocks = [
            stream_blocks[idx]
            for idx in stream_order
            if idx in stream_blocks
            and isinstance(stream_blocks[idx], dict)
            and stream_blocks[idx].get("type") == "thinking"
        ]
        # Extract text blocks from stream_blocks to ensure content_block_start
        # text is considered for has_visible_blocks even without text_delta events
        text_blocks_from_start = [
            stream_blocks[idx]
            for idx in stream_order
            if idx in stream_blocks
            and isinstance(stream_blocks[idx], dict)
            and stream_blocks[idx].get("type") == "text"
            and str(stream_blocks[idx].get("text", "")).strip()
        ]
        content_blocks = _response_contract_for_mode(full_text, tool_compatible=tool_compatible).content_blocks
        translated_blocks = thinking_blocks + content_blocks + tool_blocks + text_blocks_from_start
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
            content_blocks = thinking_blocks + processed.content_blocks + tool_blocks
            response_signals = processed.behavior_signals

            logger.info("Response mode: %s", processed.mode)
            logger.info("Response signals: %s", response_signals)
            logger.info("Content blocks count: %d", len(content_blocks))
            translated_blocks = content_blocks

        has_visible_blocks = any(
            block.get("type") == "tool_use"
            or (block.get("type") == "text" and str(block.get("text", "")).strip())
            or block.get("type") == "thinking"
            or block.get("type") == "redacted_thinking"
            for block in translated_blocks
        )
        if has_visible_blocks and all(
            block.get("type") == "text" and not str(block.get("text", "")).strip() for block in translated_blocks
        ):
            has_visible_blocks = False
        recovery_required = not has_visible_blocks and (read_error is not None or len(translated_blocks) == 0)
        _cost_recorded_by_fallback = False
        if recovery_required:
            recovery_allowed, _recovery_reason = _stream_recovery_allowed_now(session)
            stream_behavior_signals["stream_empty_after_success"] = 1
            if read_error is not None:
                stream_behavior_signals["stream_recovery_read_error"] = 1
            else:
                stream_behavior_signals["stream_recovery_empty_success"] = 1
                with contextlib.suppress(Exception):
                    session.smoothness_tracker.record(
                        SmoothnessEventType.EMPTY_STREAM_SUCCESS,
                    )
            if recovery_allowed:
                session.runtime_session._stream_recovery_reacquisition_budget = 1
                session.runtime_session._stream_recovery_history_floor_budget = 1
                session.runtime_session.note_request_policy_stream_recovery()
                session.runtime_session._stream_recovery_cooldown_remaining = 1
                session.runtime_session._stream_recovery_cooldown_suppressed = False
            else:
                session.runtime_session._stream_recovery_cooldown_suppressed = True
            recovered = False
            recovery_model = ""
            recovery_usage: dict[str, Any] = {}
            if request_content and request_url:
                stream_behavior_signals["stream_recovery_started"] = 1
                stream_behavior_signals["stream_recovery_retry"] = 1
                logger.warning(
                    "stream_recovery_retry_started: empty streamed success detected; retrying upstream non-stream"
                )
                with contextlib.suppress(Exception):
                    session.smoothness_tracker.record(
                        SmoothnessEventType.STREAM_RECOVERY_STARTED,
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
                try:
                    async with httpx.AsyncClient(timeout=300.0) as retry_client:
                        retry_request = retry_client.build_request(
                            request_method,
                            request_url,
                            headers=request_headers or {},
                            content=recovery_payload,
                        )
                        retry_response = await retry_client.send(retry_request, stream=False)
                    if retry_response.status_code == 200:
                        try:
                            retry_json = retry_response.json()
                        except Exception:
                            retry_json = None
                        if isinstance(retry_json, dict):
                            recovery_model = str(retry_json.get("model", ""))
                            recovery_usage = retry_json.get("usage", {})
                            retry_thinking_blocks = [
                                block
                                for block in retry_json.get("content", [])
                                if isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"}
                            ]
                            passthrough_blocks = [
                                block
                                for block in retry_json.get("content", [])
                                if isinstance(block, dict)
                                and block.get("type")
                                not in {
                                    "text",
                                    "thinking",
                                    "redacted_thinking",
                                }
                            ]
                            (
                                passthrough_blocks,
                                passthrough_signals,
                            ) = normalize_tool_use_blocks(
                                passthrough_blocks,
                                seed_prefix="toolu_recovery",
                            )
                            _merge_signal_counts(stream_behavior_signals, passthrough_signals)
                            retry_text = "".join(
                                str(block.get("text", ""))
                                for block in retry_json.get("content", [])
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                            retry_output_saved = 0
                            # Initialize retry_response_signals to accumulate across branches
                            retry_response_signals: dict[str, int] = {}
                            if retry_text:
                                retry_processed = _RUNTIME.process_response(
                                    retry_text,
                                    model=str(retry_json.get("model", "")),
                                    session=session.runtime_session,
                                    behavior_signals=stream_behavior_signals or None,
                                    tool_compatible=tool_compatible,
                                )
                                _merge_signal_counts(
                                    retry_response_signals,
                                    retry_processed.behavior_signals,
                                )
                                translated_blocks = (
                                    retry_thinking_blocks + retry_processed.content_blocks + passthrough_blocks
                                )
                                retry_output_saved = retry_processed.output_saved_tokens
                            elif passthrough_blocks:
                                retry_processed = _RUNTIME.process_response(
                                    "",
                                    model=str(retry_json.get("model", "")),
                                    session=session.runtime_session,
                                    behavior_signals=stream_behavior_signals or None,
                                    tool_compatible=tool_compatible,
                                )
                                _merge_signal_counts(
                                    retry_response_signals,
                                    retry_processed.behavior_signals,
                                )
                                translated_blocks = retry_thinking_blocks + passthrough_blocks
                            else:
                                # Keep response_signals as initialized from stream_behavior_signals
                                translated_blocks = []
                            recovered = any(
                                block.get("type") == "tool_use"
                                or (block.get("type") == "text" and str(block.get("text", "")).strip())
                                for block in translated_blocks
                            )
                            recovered_text = False
                            recovered_tool_use = False
                            if recovered:
                                recovered_text = any(
                                    block.get("type") == "text" and str(block.get("text", "")).strip()
                                    for block in translated_blocks
                                )
                                recovered_tool_use = any(block.get("type") == "tool_use" for block in translated_blocks)
                                if recovered_tool_use and not recovered_text:
                                    signature = _tool_use_only_signature(translated_blocks)
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
                                        stream_behavior_signals["stream_recovery_loop_broken"] = 1
                                        stream_behavior_signals["stream_recovery_fallback"] = 1
                                        logger.warning(
                                            "stream_recovery_loop_breaker_triggered: repeated identical tool_use-only recovery detected; falling back to avoid retry loop"
                                        )
                                        with contextlib.suppress(Exception):
                                            session.smoothness_tracker.record(
                                                SmoothnessEventType.STREAM_RECOVERY_LOOP_BREAKER,
                                            )
                                        recovered = False
                                else:
                                    session.runtime_session._stream_recovery_tool_use_only_signature = ""
                                    session.runtime_session._stream_recovery_tool_use_only_repeat_count = 0

                            if recovered:
                                recovery_success_signals: dict[str, int] = {}
                                if recovered_text:
                                    stream_behavior_signals["stream_recovery_success_text"] = 1
                                    recovery_success_signals["stream_recovery_success_text"] = 1
                                    logger.info(
                                        "stream_recovery_succeeded_text: recovered empty streamed success via non-stream retry"
                                    )
                                    with contextlib.suppress(Exception):
                                        session.smoothness_tracker.record(
                                            SmoothnessEventType.STREAM_RECOVERY_SUCCEEDED,
                                            {"recovery_type": "text"},
                                        )
                                if recovered_tool_use:
                                    stream_behavior_signals["stream_recovery_success_tool_use"] = 1
                                    recovery_success_signals["stream_recovery_success_tool_use"] = 1
                                    logger.info(
                                        "stream_recovery_succeeded_tool_use: recovered empty streamed success via non-stream retry"
                                    )
                                    with contextlib.suppress(Exception):
                                        session.smoothness_tracker.record(
                                            SmoothnessEventType.STREAM_RECOVERY_SUCCEEDED,
                                            {"recovery_type": "tool_use"},
                                        )
                                _merge_signal_counts(
                                    response_signals,
                                    recovery_success_signals,
                                )
                                _merge_signal_counts(
                                    response_signals,
                                    retry_response_signals,
                                )
                                retry_usage = retry_json.get("usage", {})
                                retry_model = str(retry_json.get("model", ""))
                                if retry_model and retry_usage:
                                    stream_behavior_signals["stream_recovery_usage"] = 1
                                    session.tracker.record_call(
                                        model=retry_model,
                                        actual_input=retry_usage.get("input_tokens", 0),
                                        actual_output=retry_usage.get("output_tokens", 0),
                                        cache_read=retry_usage.get("cache_read_input_tokens", 0),
                                        cache_write=retry_usage.get("cache_creation_input_tokens", 0),
                                        input_saved=input_saved_tokens,
                                        output_saved=retry_output_saved,
                                        type_breakdown=type_breakdown,
                                        behavior_signals=response_signals or None,
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
                                    for event_bytes in _emit_sse_block(i, block):
                                        yield event_bytes
                                yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
                                _clear_stream_read_error_streak(session)
                                return
                except Exception as exc:
                    recovered = False
                    stream_behavior_signals["stream_recovery_retry_exception"] = 1
                    logger.warning(
                        "stream_recovery_retry_exception: recovery retry failed: %s",
                        exc,
                    )
            if not recovered:
                stream_behavior_signals["stream_recovery_fallback"] = 1
                logger.warning(
                    "stream_recovery_fallback: non-stream retry produced no visible content; recording fallback"
                )
                if recovery_model and recovery_usage:
                    stream_behavior_signals["stream_recovery_usage"] = 1
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
                        cache_read=recovery_usage.get("cache_read_input_tokens", 0),
                        cache_write=recovery_usage.get("cache_creation_input_tokens", 0),
                        input_saved=0,
                        output_saved=0,
                        type_breakdown=type_breakdown,
                        behavior_signals=empty_processed.behavior_signals or None,
                        prompt_metrics=prompt_metrics,
                    )
                if request_state is not None:
                    _record_fallback_once(session, request_state)
                _cost_recorded_by_fallback = bool(recovery_model and recovery_usage)
        if sse_model != "unknown" and sse_usage and not _cost_recorded_by_fallback:
            if not full_text:
                processed = _RUNTIME.process_response(
                    "",
                    model=sse_model,
                    session=session.runtime_session,
                    behavior_signals=stream_behavior_signals or None,
                    tool_compatible=tool_compatible,
                )
                content_blocks = thinking_blocks + processed.content_blocks + tool_blocks
                output_saved = 0
                response_signals = processed.behavior_signals
            # processed was assigned in the outer full_text block at lines 203-209
            # Ensure we have a valid processed object before accessing attributes
            elif processed is not None:
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

                if (etype in ("message_delta", "message_stop") and not content_emitted) and content_blocks:
                    for i, block in enumerate(content_blocks):
                        for event_bytes in _emit_sse_block(i, block):
                            yield event_bytes
                    content_emitted = True

                yield (event_str + "\n\n").encode()

            except (json.JSONDecodeError, KeyError) as exc:
                logger.debug("SSE parse error (relay): %s", exc)
                yield (event_str + "\n\n").encode()
        if content_emitted:
            _clear_stream_read_error_streak(session)
    finally:
        response_aclose: Callable[[], Awaitable[None]] | None = getattr(response, "aclose", None)
        if callable(response_aclose):
            await response_aclose()
        if client_owned:
            client_aclose: Callable[[], Awaitable[None]] | None = getattr(client, "aclose", None)
            if callable(client_aclose):
                await client_aclose()
