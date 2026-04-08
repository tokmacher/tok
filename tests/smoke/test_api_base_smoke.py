"""
Custom endpoint / api-base smoke gate for Tok bridge.

This smoke test verifies:
- Explicit api-base override wins over default env-based endpoint
- No silent fallback occurs when explicit endpoint is unreachable
- Wrong-target behavior fails clearly
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from collections.abc import AsyncIterator

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from tok.gateway import BridgeSession, create_app

# -----------------------------------------------------------------------------
# Synthetic upstream state with thread-safe counters
# -----------------------------------------------------------------------------


class _DefaultUpstreamState:
    """Thread-safe state for default-path synthetic upstream."""

    def __init__(self) -> None:
        self.call_count = 0
        self.lock = asyncio.Lock()

    async def increment(self) -> int:
        async with self.lock:
            self.call_count += 1
            return self.call_count

    async def get_count(self) -> int:
        async with self.lock:
            return self.call_count

    def reset(self) -> None:
        self.call_count = 0


class _ExplicitUpstreamState:
    """Thread-safe state for explicit-override synthetic upstream."""

    def __init__(self) -> None:
        self.call_count = 0
        self.lock = asyncio.Lock()

    async def increment(self) -> int:
        async with self.lock:
            self.call_count += 1
            return self.call_count

    async def get_count(self) -> int:
        async with self.lock:
            return self.call_count

    def reset(self) -> None:
        self.call_count = 0


# Global state instances
_default_upstream_state = _DefaultUpstreamState()
_explicit_upstream_state = _ExplicitUpstreamState()


# -----------------------------------------------------------------------------
# Synthetic upstream factories with distinct markers
# -----------------------------------------------------------------------------


def _create_default_upstream(state: _DefaultUpstreamState) -> FastAPI:
    """Create default-path synthetic upstream with 'default-upstream' marker."""
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "default-upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request) -> StreamingResponse:
        await state.increment()

        async def generate_stream() -> AsyncIterator[bytes]:
            # SSE-formatted streaming response with default-upstream marker
            message_start = {
                "type": "message_start",
                "message": {
                    "model": "test-model",
                    "usage": {"input_tokens": 10},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()

            # Content block with default-upstream marker
            content_block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": "Default upstream response",
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode()

            content_block_stop = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode()

            message_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            }
            yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()

            message_stop = {"type": "message_stop"}
            yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode()

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )

    return app


def _create_explicit_upstream(state: _ExplicitUpstreamState) -> FastAPI:
    """Create explicit-override synthetic upstream with 'explicit-upstream' marker."""
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "explicit-upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request) -> StreamingResponse:
        await state.increment()

        async def generate_stream() -> AsyncIterator[bytes]:
            # SSE-formatted streaming response with explicit-upstream marker
            message_start = {
                "type": "message_start",
                "message": {
                    "model": "test-model",
                    "usage": {"input_tokens": 10},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()

            # Content block with explicit-upstream marker
            content_block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": "Explicit upstream response",
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode()

            content_block_stop = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode()

            message_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            }
            yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()

            message_stop = {"type": "message_stop"}
            yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode()

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )

    return app


# -----------------------------------------------------------------------------
# Server startup helpers
# -----------------------------------------------------------------------------


async def _start_upstream_server(app: FastAPI, port: int) -> asyncio.Task:
    """Start synthetic upstream server on specified port."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    # Wait for server to start
    for _ in range(50):  # 5 seconds max wait
        await asyncio.sleep(0.1)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://127.0.0.1:{port}/")
                if resp.status_code == 200:
                    break
        except Exception:
            pass
    else:
        task.cancel()
        raise RuntimeError(f"Upstream server failed to start on port {port}")

    return task


def _find_free_port() -> int:
    """Find an ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -----------------------------------------------------------------------------
# Positive override test: explicit api-base wins over default env
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_api_base_wins_over_default_env() -> None:
    """
    Positive override smoke test.

    Verifies:
    - Explicit api_base parameter is honored over TOK_API_BASE env var
    - Default env-based endpoint is NOT used when explicit override is set
    - Request flows to the explicit upstream, not the default upstream
    """
    global _default_upstream_state, _explicit_upstream_state

    # Reset counters before test
    _default_upstream_state.reset()
    _explicit_upstream_state.reset()

    # Find ephemeral ports
    default_port = _find_free_port()
    explicit_port = _find_free_port()
    bridge_port = _find_free_port()

    # Start both upstream servers
    default_app = _create_default_upstream(_default_upstream_state)
    explicit_app = _create_explicit_upstream(_explicit_upstream_state)

    default_task = await _start_upstream_server(default_app, default_port)
    explicit_task = await _start_upstream_server(explicit_app, explicit_port)

    default_base = f"http://127.0.0.1:{default_port}"
    explicit_base = f"http://127.0.0.1:{explicit_port}"

    # Save original env
    orig_api_base = os.environ.get("TOK_API_BASE")

    try:
        # Set TOK_API_BASE to default upstream (should be ignored by explicit override)
        os.environ["TOK_API_BASE"] = default_base

        # Create bridge session with EXPLICIT override pointing at explicit upstream
        bridge_session = BridgeSession(
            port=bridge_port,
            api_base=explicit_base,  # This should win over env var
            debug=False,
            fail_open=False,
        )

        # Create and start bridge server
        bridge_app = create_app(bridge_session)
        bridge_config = uvicorn.Config(
            bridge_app,
            host="127.0.0.1",
            port=bridge_port,
            log_level="error",
        )
        bridge_server = uvicorn.Server(bridge_config)
        bridge_task = asyncio.create_task(bridge_server.serve())

        # Wait for bridge to start
        for _ in range(50):
            await asyncio.sleep(0.1)
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{bridge_port}/health")
                    if resp.status_code == 200:
                        break
            except Exception:
                pass
        else:
            bridge_task.cancel()
            raise RuntimeError(f"Bridge server failed to start on port {bridge_port}")

        try:
            # Send streaming request through bridge
            request_body = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello, explicit!"}],
                "stream": True,
            }

            collected_chunks: list[bytes] = []
            default_before = await _default_upstream_state.get_count()
            explicit_before = await _explicit_upstream_state.get_count()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{bridge_port}/v1/messages",
                    json=request_body,
                    headers={"x-api-key": "test-api-key"},
                    timeout=30.0,
                )

                # Collect all streamed chunks
                async for chunk in response.aiter_bytes():
                    collected_chunks.append(chunk)

            default_after = await _default_upstream_state.get_count()
            explicit_after = await _explicit_upstream_state.get_count()
            default_calls = default_after - default_before
            explicit_calls = explicit_after - explicit_before

            full_response = b"".join(collected_chunks).decode("utf-8", errors="replace")

            # CRITICAL: Explicit upstream must be called exactly once
            assert explicit_calls == 1, (
                f"EXPLICIT API BASE NOT HONORED: explicit upstream called {explicit_calls} times, "
                f"expected exactly 1. The explicit api_base parameter may be ignored."
            )

            # CRITICAL: Default upstream must NOT be called (no silent fallback)
            assert default_calls == 0, (
                f"SILENT FALLBACK TO DEFAULT API BASE: default upstream called {default_calls} times, "
                f"expected 0. The bridge fell back to TOK_API_BASE despite explicit override."
            )

            # CRITICAL: Response must come from explicit upstream
            assert "Explicit upstream response" in full_response, (
                f"WRONG TARGET RESPONSE: expected explicit upstream marker in response. "
                f"Response was: {full_response[:500]}"
            )

            # CRITICAL: Response must NOT contain default upstream marker
            assert "Default upstream response" not in full_response, (
                "MULTIPLE TARGETS OBSERVED: default upstream marker found in response. "
                "Both upstreams may have been called."
            )

        finally:
            # Stop bridge server
            bridge_server.should_exit = True
            try:
                await asyncio.wait_for(bridge_task, timeout=5.0)
            except asyncio.TimeoutError:
                bridge_task.cancel()

    finally:
        # Restore original env
        if orig_api_base is None:
            os.environ.pop("TOK_API_BASE", None)
        else:
            os.environ["TOK_API_BASE"] = orig_api_base

        # Stop upstream servers
        default_task.cancel()
        explicit_task.cancel()
        try:
            await asyncio.wait_for(default_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        try:
            await asyncio.wait_for(explicit_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # Final verification
    final_explicit = await _explicit_upstream_state.get_count()
    final_default = await _default_upstream_state.get_count()

    assert final_explicit == 1, f"UPSTREAM CALL MISMATCH: explicit upstream expected 1 call, got {final_explicit}"
    assert final_default == 0, f"UPSTREAM CALL MISMATCH: default upstream expected 0 calls, got {final_default}"


# -----------------------------------------------------------------------------
# Negative no-fallback test: unreachable explicit fails clearly
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreachable_explicit_endpoint_fails_clearly() -> None:
    """
    Negative no-fallback smoke test.

    Verifies:
    - When explicit api_base is unreachable, request FAILS (does not fall back)
    - Default env-based endpoint is NOT used as fallback
    - Wrong-target behavior produces clear failure
    """
    global _default_upstream_state

    # Reset counter before test
    _default_upstream_state.reset()

    # Find ephemeral ports
    default_port = _find_free_port()
    bridge_port = _find_free_port()

    # Use a definitely-unused port for unreachable explicit endpoint
    # Verify it's unused first
    unreachable_port = _find_free_port()
    while unreachable_port in (default_port, bridge_port):
        unreachable_port = _find_free_port()

    # Start only the default upstream
    default_app = _create_default_upstream(_default_upstream_state)
    default_task = await _start_upstream_server(default_app, default_port)

    default_base = f"http://127.0.0.1:{default_port}"
    unreachable_base = f"http://127.0.0.1:{unreachable_port}"

    # Save original env
    orig_api_base = os.environ.get("TOK_API_BASE")

    try:
        # Set TOK_API_BASE to working default upstream
        os.environ["TOK_API_BASE"] = default_base

        # Create bridge session with UNREACHABLE explicit override
        bridge_session = BridgeSession(
            port=bridge_port,
            api_base=unreachable_base,  # This endpoint does not exist
            debug=False,
            fail_open=False,
        )

        # Create and start bridge server
        bridge_app = create_app(bridge_session)
        bridge_config = uvicorn.Config(
            bridge_app,
            host="127.0.0.1",
            port=bridge_port,
            log_level="error",
        )
        bridge_server = uvicorn.Server(bridge_config)
        bridge_task = asyncio.create_task(bridge_server.serve())

        # Wait for bridge to start
        for _ in range(50):
            await asyncio.sleep(0.1)
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{bridge_port}/health")
                    if resp.status_code == 200:
                        break
            except Exception:
                pass
        else:
            bridge_task.cancel()
            raise RuntimeError(f"Bridge server failed to start on port {bridge_port}")

        try:
            # Send streaming request through bridge - should FAIL
            request_body = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello, unreachable!"}],
                "stream": True,
            }

            default_before = await _default_upstream_state.get_count()

            request_failed = False
            failure_reason = ""

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://127.0.0.1:{bridge_port}/v1/messages",
                        json=request_body,
                        headers={"x-api-key": "test-api-key"},
                        timeout=10.0,
                    )

                    # If we get here, check if it's an error response
                    if response.status_code != 200:
                        request_failed = True
                        failure_reason = f"HTTP {response.status_code}"
                    else:
                        # Unexpected success - check what we got
                        content = await response.aread()
                        failure_reason = f"Unexpected 200 response: {content[:200]}"

            except (httpx.ConnectError, httpx.RequestError) as e:
                request_failed = True
                failure_reason = f"{type(e).__name__}: {e}"

            default_after = await _default_upstream_state.get_count()
            default_calls = default_after - default_before

            # CRITICAL: Request MUST fail (not silently succeed)
            assert request_failed, (
                f"WRONG-TARGET FAILURE DID NOT FAIL CLEARLY: request succeeded when it should have "
                f"failed. Unreachable explicit endpoint at {unreachable_base} was expected to "
                f"cause failure. {failure_reason}"
            )

            # CRITICAL: Default upstream must NOT be called (no fallback)
            assert default_calls == 0, (
                f"UNEXPECTED FALLBACK TO DEFAULT API BASE: default upstream called {default_calls} "
                f"times when explicit endpoint was unreachable. Silent fallback occurred."
            )

        finally:
            # Stop bridge server
            bridge_server.should_exit = True
            try:
                await asyncio.wait_for(bridge_task, timeout=5.0)
            except asyncio.TimeoutError:
                bridge_task.cancel()

    finally:
        # Restore original env
        if orig_api_base is None:
            os.environ.pop("TOK_API_BASE", None)
        else:
            os.environ["TOK_API_BASE"] = orig_api_base

        # Stop upstream server
        default_task.cancel()
        try:
            await asyncio.wait_for(default_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # Final verification: default upstream was never called
    final_default = await _default_upstream_state.get_count()
    assert final_default == 0, f"UPSTREAM CALL MISMATCH: default upstream expected 0 calls, got {final_default}"
