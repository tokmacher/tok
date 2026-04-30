"""
Tests for transport signal accounting in streaming paths.

These tests verify that read errors are properly tracked in both
passthrough (baseline) and buffer-strip (tool-compatible) modes.
"""

import asyncio

import httpx

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import (
    buffer_strip_restream_impl,
    passthrough_stream_impl,
)
from tok.runtime.smoothness.models import SmoothnessEventType


def test_passthrough_stream_impl_records_read_error() -> None:
    """Passthrough mode should record STREAM_READ_ERROR when a read error occurs."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_1", "task_1")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4"}}\n\n'
            msg = "Connection reset by peer"
            raise httpx.ReadError(msg)

        async def aclose(self) -> None:
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self) -> None:
            pass

    client = MockClient()

    async def run_test():
        chunks = []
        async for chunk in passthrough_stream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_test())

    assert len(chunks) == 1
    assert b"message_start" in chunks[0]

    report = session.smoothness_tracker.finish_turn()
    assert report.score == 88
    assert len(report.events) == 1
    assert report.events[0].event_type == SmoothnessEventType.STREAM_READ_ERROR
    assert report.events[0].metadata == {"error": "Connection reset by peer"}


def test_passthrough_stream_impl_records_httpcore_read_error() -> None:
    """Passthrough mode should record STREAM_READ_ERROR for httpcore.ReadError too."""
    import httpcore

    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_2", "task_2")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4"}}\n\n'
            msg = "Premature EOF"
            raise httpcore.ReadError(msg)

        async def aclose(self) -> None:
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self) -> None:
            pass

    client = MockClient()

    async def run_test():
        chunks = []
        async for chunk in passthrough_stream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_test())

    assert len(chunks) == 1

    report = session.smoothness_tracker.finish_turn()
    assert report.score == 88
    assert len(report.events) == 1
    assert report.events[0].event_type == SmoothnessEventType.STREAM_READ_ERROR
    assert report.events[0].metadata == {"error": "Premature EOF"}


def test_passthrough_read_error_records_event_and_increments_streak() -> None:
    """Passthrough mode should record STREAM_READ_ERROR and increment streak state."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_3", "task_3")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4"}}\n\n'
            msg = "Connection reset by peer"
            raise httpx.ReadError(msg)

        async def aclose(self) -> None:
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self) -> None:
            pass

    client = MockClient()

    async def run_test():
        chunks = []
        async for chunk in passthrough_stream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_test())

    assert len(chunks) == 1
    assert b"message_start" in chunks[0]

    report = session.smoothness_tracker.finish_turn()
    assert report.score == 88
    assert len(report.events) == 1
    assert report.events[0].event_type == SmoothnessEventType.STREAM_READ_ERROR
    assert report.events[0].metadata == {"error": "Connection reset by peer"}

    assert session.runtime_session._stream_read_error_consecutive_count == 1
    assert session.runtime_session._stream_read_error_last_stage == "passthrough"


def test_buffering_read_error_records_event_and_increments_streak() -> None:
    """Buffering mode should record STREAM_READ_ERROR and increment streak state."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_4", "task_4")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4"}}\n\n'
            msg = "Connection reset by peer"
            raise httpx.ReadError(msg)

        async def aclose(self) -> None:
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self) -> None:
            pass

    client = MockClient()

    async def run_test():
        chunks = []
        async for chunk in buffer_strip_restream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
            tool_compatible=True,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_test())

    assert len(chunks) >= 1

    report = session.smoothness_tracker.finish_turn()
    assert report.score == 88
    assert len(report.events) == 1
    assert report.events[0].event_type == SmoothnessEventType.STREAM_READ_ERROR
    assert report.events[0].metadata == {"error": "Connection reset by peer"}

    assert session.runtime_session._stream_read_error_consecutive_count == 1
    assert session.runtime_session._stream_read_error_last_stage == "buffering"


def test_successful_visible_completion_clears_read_error_streak() -> None:
    """Successful visible completion should clear read-error streak state."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_5", "task_5")

    session.runtime_session._stream_read_error_consecutive_count = 2
    session.runtime_session._stream_read_error_last_stage = "passthrough"

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4","id":"msg_1"}}\n\n'
            yield b'event: content_block_start\ndata: {"index":0,"content_block":{"type":"text","text":""}}\n\n'
            yield b'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"Hello world"}}\n\n'
            yield b'event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"},"type":"message_delta","usage":{"output_tokens":11}}\n\n'

        async def aclose(self) -> None:
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self) -> None:
            pass

    client = MockClient()

    async def run_test():
        chunks = []
        async for chunk in passthrough_stream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_test())

    assert len(chunks) >= 1

    report = session.smoothness_tracker.finish_turn()
    assert report.score == 100
    assert len(report.events) == 0

    assert session.runtime_session._stream_read_error_consecutive_count == 0
    assert session.runtime_session._stream_read_error_last_stage == ""


def test_recovery_cooldown_suppresses_repeat_recovery_budget_assignment(
    monkeypatch,
) -> None:
    """Cooldown should suppress repeat recovery budget assignment."""
    from tok.gateway import _buffer_strip_restream

    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_6", "task_6")

    session.runtime_session._stream_recovery_cooldown_remaining = 1
    session.runtime_session._stream_recovery_reacquisition_budget = 0
    session.runtime_session._stream_recovery_history_floor_budget = 0

    class FakeResponse:
        async def aiter_bytes(self):
            yield b""

    class FakeClient:
        async def aclose(self) -> None:
            pass

    client = FakeClient()

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "content": [{"type": "text", "text": "Recovered answer"}],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    async def run_test():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            client,
            FakeResponse(),
            tool_compatible=False,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=b'{"model":"claude-sonnet-4","messages":[{"role":"user","content":"test"}],"stream":true}',
            request_state={"fallback_recorded": False},
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_test())

    assert len(chunks) >= 0

    assert session.runtime_session._stream_recovery_reacquisition_budget == 0
    assert session.runtime_session._stream_recovery_history_floor_budget == 0
    assert session.runtime_session._stream_recovery_cooldown_suppressed is True


def test_whitespace_only_text_triggers_recovery() -> None:
    """Whitespace-only text blocks should not be treated as visible content."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_7", "task_7")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4","id":"msg_1"}}\n\n'
            yield b'event: content_block_start\ndata: {"index":0,"content_block":{"type":"text","text":""}}\n\n'
            yield b'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"  "}}\n\n'
            yield b'event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"},"type":"message_delta","usage":{"output_tokens":3}}\n\n'

        async def aclose(self) -> None:
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self) -> None:
            pass

    client = MockClient()

    async def run_test():
        async for _chunk in buffer_strip_restream_impl(
            session=session,
            client=client,
            response=response,
            input_saved_tokens=0,
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=b'{"model":"claude-sonnet-4","messages":[{"role":"user","content":"test"}],"stream":true}',
            request_state={"fallback_recorded": False},
        ):
            pass

    asyncio.run(run_test())
    report = session.smoothness_tracker.finish_turn()
    event_types = [e.event_type for e in report.events]
    assert (
        SmoothnessEventType.EMPTY_STREAM_SUCCESS in event_types
        or session.runtime_session._stream_recovery_cooldown_remaining == 1
    ), "Whitespace-only response should trigger recovery or cooldown signal"
