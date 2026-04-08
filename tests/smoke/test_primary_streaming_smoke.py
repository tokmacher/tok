"""
Primary streaming smoke gate for Tok bridge.

This smoke test verifies:
- Tok accepts a real streaming request on the supported path
- Tok forwards it to a controlled upstream target
- Streamed chunks actually flow through the bridge
- The stream completes cleanly
- The upstream is hit exactly once
- The smoke fails loudly if streaming breaks
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from tok.gateway import BridgeSession, create_app

# -----------------------------------------------------------------------------
# Synthetic upstream with call counter
# -----------------------------------------------------------------------------


class _UpstreamState:
    """Thread-safe state for synthetic upstream."""

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


# Global state for the synthetic upstream
_upstream_state = _UpstreamState()


def _create_synthetic_upstream(state: _UpstreamState) -> FastAPI:
    """Create a synthetic upstream that emits deterministic streaming chunks."""
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request) -> StreamingResponse:
        await state.increment()

        async def generate_stream() -> AsyncIterator[bytes]:
            # SSE-formatted streaming response with content that survives Tok processing
            message_start = {
                "type": "message_start",
                "message": {
                    "model": "test-model",
                    "usage": {"input_tokens": 10},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()

            # First content block - use text that looks like natural response
            content_block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "First part "},
            }
            yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode()

            # Second content block delta
            content_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "second part"},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n".encode()

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


async def _start_upstream_server(state: _UpstreamState, port: int) -> asyncio.Task:
    """Start the synthetic upstream server on an ephemeral port."""
    app = _create_synthetic_upstream(state)
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


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_streaming_smoke() -> None:
    """
    Primary streaming smoke gate.

    Verifies:
    - Bridge accepts real streaming request
    - Chunks flow through bridge to upstream
    - Stream completes cleanly
    - Upstream is called exactly once
    - Fails loudly if streaming breaks
    """
    # Reset call counter before test
    global _upstream_state
    _upstream_state.reset()

    # Find ephemeral ports
    import socket

    def find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    upstream_port = find_free_port()
    bridge_port = find_free_port()

    # Start synthetic upstream server
    upstream_task = await _start_upstream_server(_upstream_state, upstream_port)
    upstream_base = f"http://127.0.0.1:{upstream_port}"

    try:
        # Create bridge session pointing at our synthetic upstream
        bridge_session = BridgeSession(
            port=bridge_port,
            api_base=upstream_base,
            debug=False,
            fail_open=False,  # Fail closed so we see real errors
        )

        # Create the bridge app
        bridge_app = create_app(bridge_session)

        # Start bridge server
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
            upstream_task.cancel()
            raise RuntimeError(f"Bridge server failed to start on port {bridge_port}")

        try:
            # Send a real streaming request through the bridge
            request_body = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello, stream!"}],
                "stream": True,
            }

            # Execute streaming request
            collected_chunks: list[bytes] = []
            upstream_call_before = await _upstream_state.get_count()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{bridge_port}/v1/messages",
                    json=request_body,
                    headers={"x-api-key": "test-api-key"},
                    timeout=30.0,
                )

                # Verify response status
                assert response.status_code == 200, (
                    f"Streaming request failed with status {response.status_code}: {response.text}"
                )

                # Collect all streamed chunks
                async for chunk in response.aiter_bytes():
                    collected_chunks.append(chunk)

            # Verify upstream call count increased by exactly 1
            upstream_call_after = await _upstream_state.get_count()
            actual_calls = upstream_call_after - upstream_call_before

            # CRITICAL: Fail loudly if duplicate calls detected
            assert actual_calls == 1, (
                f"DUPLICATE EXECUTION DETECTED: upstream was called {actual_calls} times, "
                f"expected exactly 1. This indicates a serious bug where the bridge "
                f"silently retries or double-calls the upstream."
            )

            # CRITICAL: Fail loudly if no upstream call (wrong target)
            assert actual_calls > 0, (
                f"WRONG TARGET OR NO CALL: upstream call count is {actual_calls}, "
                f"expected exactly 1. The bridge may be targeting a different endpoint "
                f"or silently failing."
            )

            # Verify we received chunks
            assert len(collected_chunks) > 0, (
                "STREAM FAILURE: No chunks were received from the streaming response. "
                "The stream may have been empty or the connection may have failed silently."
            )

            # Verify the complete response contains expected markers
            full_response = b"".join(collected_chunks)
            response_text = full_response.decode("utf-8", errors="replace")

            # Verify SSE event structure is present (most important: stream format valid)
            assert "event: message_start" in response_text, (
                f"INVALID STREAM FORMAT: missing 'message_start' event. Response was: {response_text[:500]}"
            )
            assert "event: message_stop" in response_text, (
                f"INVALID STREAM FORMAT: missing 'message_stop' event. Response was: {response_text[:500]}"
            )

            # Verify content was received (check for content events, not exact text)
            # Note: Tok may transform content, but events should flow through
            assert "content_block" in response_text, (
                f"MISSING CONTENT: no content_block events in response. Response was: {response_text[:500]}"
            )

        finally:
            # Stop bridge server
            bridge_server.should_exit = True
            try:
                await asyncio.wait_for(bridge_task, timeout=5.0)
            except asyncio.TimeoutError:
                bridge_task.cancel()

    finally:
        # Stop upstream server
        upstream_task.cancel()
        try:
            await asyncio.wait_for(upstream_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # Final verification that exactly one call was made total
    final_call_count = await _upstream_state.get_count()
    assert final_call_count == 1, (
        f"UPSTREAM CALL MISMATCH: expected exactly 1 call, got {final_call_count}. "
        f"This indicates the bridge made unexpected additional calls."
    )
