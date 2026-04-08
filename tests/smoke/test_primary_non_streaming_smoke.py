"""
Primary non-streaming smoke gate for Tok bridge.

This smoke test verifies:
- Tok accepts a real non-streaming request on the supported path
- Tok forwards it to a controlled upstream target
- Complete response payload flows through the bridge (not chunked)
- The response completes cleanly
- The upstream is hit exactly once
- Fails loudly if non-streaming path breaks
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
    """Create a synthetic upstream that returns a complete non-streaming response."""
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request) -> JSONResponse:
        await state.increment()

        # Complete non-streaming JSON response
        response_data = {
            "id": "msg_12345",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This is a complete non-streaming response.",
                }
            ],
            "model": "test-model",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }

        return JSONResponse(content=response_data)

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
async def test_primary_non_streaming_smoke() -> None:
    """
    Primary non-streaming smoke gate.

    Verifies:
    - Bridge accepts real non-streaming request
    - Complete payload flows through bridge to upstream
    - Response completes cleanly
    - Upstream is called exactly once
    - Fails loudly if non-streaming path breaks
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
            # Send a real non-streaming request through the bridge
            request_body = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello, complete response!"}],
                "stream": False,  # NON-STREAMING
            }

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
                    f"Non-streaming request failed with status {response.status_code}: {response.text}"
                )

                # Get complete response (non-streaming means single payload)
                response_data = response.json()

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

            # Verify response is valid JSON with expected structure
            assert "content" in response_data, (
                f"INVALID RESPONSE: missing 'content' field. Response was: {json.dumps(response_data)[:500]}"
            )

            # Verify it's a complete message response (not streaming chunks)
            assert response_data.get("type") == "message", (
                "INVALID RESPONSE TYPE: expected 'message' for non-streaming, "
                f"got '{response_data.get('type')}'. Response was: {json.dumps(response_data)[:500]}"
            )

            # Verify the response contains actual content
            content = response_data.get("content", [])
            assert len(content) > 0, (
                f"MISSING CONTENT: response content array is empty. Response was: {json.dumps(response_data)[:500]}"
            )

            # Verify usage stats are present (complete response marker)
            assert "usage" in response_data, (
                "MISSING USAGE: non-streaming response should include usage stats. "
                f"Response was: {json.dumps(response_data)[:500]}"
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
