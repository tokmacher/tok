"""Automated live-smoke matrix for Claude-through-Tok exercised boundaries."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tok.gateway import BridgeSession, create_app

_HOST = "127.0.0.1"
_WAIT_RETRIES = 50
_WAIT_SECONDS = 0.1
_REQUEST_TIMEOUT_SECONDS = 30.0
_LONG_PROMPT_CHARS = 24_000


class _UpstreamState:
    """Thread-safe state used by synthetic upstream handlers."""

    def __init__(self) -> None:
        self.call_count = 0
        self.lock = asyncio.Lock()
        self.last_request: dict[str, Any] | None = None

    async def increment(self, request_body: dict[str, Any]) -> int:
        async with self.lock:
            self.call_count += 1
            self.last_request = request_body
            return self.call_count

    async def get_count(self) -> int:
        async with self.lock:
            return self.call_count

    async def get_last_request(self) -> dict[str, Any] | None:
        async with self.lock:
            return self.last_request


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return sock.getsockname()[1]


def _create_upstream(state: _UpstreamState) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "live-smoke-upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request):
        payload = await request.json()
        await state.increment(payload)

        if bool(payload.get("stream")):

            async def generate_stream() -> AsyncIterator[bytes]:
                yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
                yield (
                    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
                    b'"content_block":{"type":"text","text":"stream smoke"}}\n\n'
                )
                yield (b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n')
                yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # Emit tool-use block when caller asks for tool category coverage.
        if payload.get("metadata", {}).get("smoke_category") == "tool-use":
            return JSONResponse(
                content={
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_smoke_1",
                            "name": "view_file",
                            "input": {"path": "README.md"},
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                }
            )

        return JSONResponse(
            content={
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }
        )

    return app


async def _start_server(app: FastAPI, port: int, health_path: str = "/") -> tuple[uvicorn.Server, asyncio.Task]:
    config = uvicorn.Config(app, host=_HOST, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    for _ in range(_WAIT_RETRIES):
        await asyncio.sleep(_WAIT_SECONDS)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://{_HOST}:{port}{health_path}")
                if resp.status_code == 200:
                    return server, task
        except Exception:
            pass

    task.cancel()
    raise RuntimeError(f"server failed to start on port {port}")


async def _stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    with suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)


async def _start_bridge(api_base: str, bridge_port: int) -> tuple[uvicorn.Server, asyncio.Task]:
    session = BridgeSession(port=bridge_port, api_base=api_base, debug=False, fail_open=False)
    bridge_app = create_app(session)
    return await _start_server(bridge_app, bridge_port, health_path="/health")


@pytest.mark.asyncio
async def test_live_smoke_tool_use_path() -> None:
    """Pass criterion: tool_use response survives bridge and upstream call count is exactly one."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "list files"}],
            "metadata": {"smoke_category": "tool-use"},
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-api-key"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["content"][0]["type"] == "tool_use"
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_live_smoke_long_context_path() -> None:
    """Pass criterion: long prompt reaches upstream once and bridge returns success without malformed error."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        long_prompt = "A" * _LONG_PROMPT_CHARS
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": long_prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-api-key"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        assert response.status_code == 200
        upstream_payload = await state.get_last_request()
        assert upstream_payload is not None
        assert len(json.dumps(upstream_payload)) >= _LONG_PROMPT_CHARS
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_live_smoke_repeated_call_guard() -> None:
    """Pass criterion: two identical requests produce exactly two upstream calls (no hidden retries)."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "repeat"}],
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            for _ in range(2):
                response = await client.post(
                    f"http://{_HOST}:{bridge_port}/v1/messages",
                    json=request_body,
                    headers={"x-api-key": "test-api-key"},
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                assert response.status_code == 200
        assert await state.get_count() == 2
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_live_smoke_stream_cleanup_early_close() -> None:
    """Pass criterion: early client close does not trigger duplicate upstream execution."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "stream then close"}],
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-api-key"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                assert response.status_code == 200
                async for _chunk in response.aiter_bytes():
                    break
        await asyncio.sleep(0.2)
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)
