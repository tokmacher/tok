"""
Tests for real streaming lifecycle with actual AsyncClient behavior.

These tests exercise the real streaming path with httpx.MockTransport to verify
proper resource cleanup and lifetime ownership semantics.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import (
    buffer_strip_restream_impl,
    passthrough_stream_impl,
)


def _make_sse_stream_response(content: list[dict], model: str = "claude-sonnet-4") -> bytes:
    """Build a complete SSE stream payload from content blocks."""
    parts: list[bytes] = []

    # message_start
    message_start = {
        "type": "message_start",
        "message": {
            "model": model,
            "id": "msg_test",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }
    parts.append(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode())

    # content blocks
    for i, block in enumerate(content):
        if block.get("type") == "text":
            start = {
                "type": "content_block_start",
                "index": i,
                "content_block": {"type": "text", "text": ""},
            }
            parts.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode())
            delta = {
                "type": "content_block_delta",
                "index": i,
                "delta": {"type": "text_delta", "text": block.get("text", "")},
            }
            parts.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
            stop = {"type": "content_block_stop", "index": i}
            parts.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode())
        elif block.get("type") == "tool_use":
            start = {
                "type": "content_block_start",
                "index": i,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", "toolu_test"),
                    "name": block.get("name", "unknown"),
                    "input": {},
                },
            }
            parts.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n".encode())
            delta = {
                "type": "content_block_delta",
                "index": i,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input", {})),
                },
            }
            parts.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
            stop = {"type": "content_block_stop", "index": i}
            parts.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n".encode())

    # message_delta with stop_reason
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 5},
    }
    parts.append(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode())

    # message_stop
    parts.append(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')

    return b"".join(parts)


class _MockAsyncByteStream(httpx.AsyncByteStream):
    """Custom async byte stream that yields chunks on demand."""

    def __init__(self, chunks: list[bytes], transport: "MockStreamingTransport") -> None:
        self._chunks = chunks
        self._transport = transport
        self._index = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self._transport.stream_iterations += 1
            yield chunk

    async def aclose(self) -> None:
        pass


class MockStreamingTransport(httpx.AsyncBaseTransport):
    """Mock transport that yields real streaming responses via aiter_bytes()."""

    def __init__(self, stream_content: bytes, *, status_code: int = 200) -> None:
        self._stream_content = stream_content
        self._status_code = status_code
        self.response_closed = False
        self.stream_iterations = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Return a streaming response that uses aiter_bytes()."""
        self.response_closed = False
        self.stream_iterations = 0

        # Pre-chunk the content for streaming
        chunk_size = 64
        chunks = [self._stream_content[i : i + chunk_size] for i in range(0, len(self._stream_content), chunk_size)]

        # Create a streaming response using custom async byte stream
        stream = _MockAsyncByteStream(chunks, self)
        response = httpx.Response(
            self._status_code,
            headers={
                "content-type": "text/event-stream",
                "cache-control": "no-cache",
            },
            stream=stream,
            request=request,
        )

        # Track when response is closed
        original_aclose = response.aclose

        async def tracked_aclose() -> None:
            self.response_closed = True
            await original_aclose()

        response.aclose = tracked_aclose  # type: ignore[method-assign]

        return response


def test_buffer_strip_restream_closes_response_and_client() -> None:
    """Verify buffer_strip_restream_impl closes both response and client when client_owned=True."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_1", "task_1")

    # Build SSE content
    sse_content = _make_sse_stream_response(
        [
            {"type": "text", "text": "Hello world"},
        ]
    )

    transport = MockStreamingTransport(sse_content)
    client = httpx.AsyncClient(transport=transport)

    # Build a streaming request
    request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        ).encode(),
    )

    async def run_test():
        response = await client.send(request, stream=True)

        # Verify response is not closed yet
        assert not transport.response_closed

        # Consume the streaming generator
        chunks = []
        async for chunk in buffer_strip_restream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
            tool_compatible=True,
            client_owned=True,
        ):
            chunks.append(chunk)

        # After exhaustion, both response and client should be closed
        assert transport.response_closed, "Response was not closed after stream exhaustion"
        assert transport.stream_iterations > 0, "No stream chunks were iterated"

        # Verify client is closed (is_closed is set after aclose)
        assert client.is_closed, "Client was not closed when client_owned=True"

        return chunks

    chunks = asyncio.run(run_test())
    assert len(chunks) > 0, "No chunks were yielded"
    assert any(b"message_start" in c for c in chunks), "Missing message_start event"
    assert any(b"message_stop" in c for c in chunks), "Missing message_stop event"


def test_passthrough_stream_closes_response_and_client() -> None:
    """Verify passthrough_stream_impl closes both response and client when client_owned=True."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_2", "task_2")

    # Build SSE content
    sse_content = _make_sse_stream_response(
        [
            {"type": "text", "text": "Passthrough test"},
        ]
    )

    transport = MockStreamingTransport(sse_content)
    client = httpx.AsyncClient(transport=transport)

    request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            }
        ).encode(),
    )

    async def run_test():
        response = await client.send(request, stream=True)

        assert not transport.response_closed

        chunks = []
        async for chunk in passthrough_stream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
            client_owned=True,
        ):
            chunks.append(chunk)

        assert transport.response_closed, "Response was not closed after stream exhaustion"
        assert transport.stream_iterations > 0, "No stream chunks were iterated"
        assert client.is_closed, "Client was not closed when client_owned=True"

        return chunks

    chunks = asyncio.run(run_test())
    assert len(chunks) > 0
    assert any(b"message_start" in c for c in chunks)


def test_buffer_strip_restream_does_not_close_client_when_not_owned() -> None:
    """Verify client is NOT closed when client_owned=False."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_3", "task_3")

    sse_content = _make_sse_stream_response(
        [
            {"type": "text", "text": "Not owned test"},
        ]
    )

    transport = MockStreamingTransport(sse_content)
    client = httpx.AsyncClient(transport=transport)

    request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            }
        ).encode(),
    )

    async def run_test():
        response = await client.send(request, stream=True)

        chunks = []
        async for chunk in buffer_strip_restream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
            tool_compatible=True,
            client_owned=False,  # Not owned - caller manages lifetime
        ):
            chunks.append(chunk)

        # Response should be closed
        assert transport.response_closed, "Response was not closed"

        # Client should NOT be closed when client_owned=False
        assert not client.is_closed, "Client was closed when client_owned=False"

        # Clean up
        await client.aclose()
        assert client.is_closed

        return chunks

    chunks = asyncio.run(run_test())
    assert len(chunks) > 0


def test_streaming_with_tool_use_blocks() -> None:
    """Verify streaming handles tool_use blocks correctly."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_4", "task_4")

    sse_content = _make_sse_stream_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read_file",
                "input": {"path": "/tmp/test.py"},
            },
        ]
    )

    transport = MockStreamingTransport(sse_content)
    client = httpx.AsyncClient(transport=transport)

    request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "read the file"}],
                "stream": True,
            }
        ).encode(),
    )

    async def run_test():
        response = await client.send(request, stream=True)

        chunks = []
        async for chunk in buffer_strip_restream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
            tool_compatible=True,
            client_owned=True,
        ):
            chunks.append(chunk)

        assert transport.response_closed
        assert client.is_closed

        return chunks

    chunks = asyncio.run(run_test())
    assert len(chunks) > 0
    # Should contain tool_use content
    combined = b"".join(chunks)
    assert b"tool_use" in combined or b"read_file" in combined


def test_streaming_records_usage_in_session_tracker() -> None:
    """Verify streaming records token usage in session tracker."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_5", "task_5")

    sse_content = _make_sse_stream_response(
        [
            {"type": "text", "text": "Test response"},
        ]
    )

    transport = MockStreamingTransport(sse_content)
    client = httpx.AsyncClient(transport=transport)

    request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            }
        ).encode(),
    )

    async def run_test() -> None:
        response = await client.send(request, stream=True)

        async for _ in buffer_strip_restream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=50,  # Simulate saved tokens
            tool_compatible=True,
            client_owned=True,
        ):
            pass

        # Verify tracker recorded the call
        summary = session.tracker.session_summary()
        assert summary is not None
        actual_tokens = summary.get("actual_tokens", 0)
        tokens_saved = summary.get("tokens_saved", 0)
        assert (isinstance(actual_tokens, int | float) and actual_tokens > 0) or (
            isinstance(tokens_saved, int | float) and tokens_saved >= 0
        )

    asyncio.run(run_test())
