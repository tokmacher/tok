"""Tests for transport signal accounting in streaming paths.

These tests verify that read errors are properly tracked in both
passthrough (baseline) and buffer-strip (tool-compatible) modes.
"""

import asyncio

import httpx

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import passthrough_stream_impl
from tok.runtime.smoothness.models import SmoothnessEventType


def test_passthrough_stream_impl_records_read_error():
    """Passthrough mode should record STREAM_READ_ERROR when a read error occurs."""
    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_1", "task_1")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4"}}\n\n'
            raise httpx.ReadError("Connection reset by peer")

        async def aclose(self):
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self):
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


def test_passthrough_stream_impl_records_httpcore_read_error():
    """Passthrough mode should record STREAM_READ_ERROR for httpcore.ReadError too."""
    import httpcore

    session = BridgeSession()
    session.smoothness_tracker.start_turn("turn_2", "task_2")

    class MockStreamResponse:
        async def aiter_bytes(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4"}}\n\n'
            raise httpcore.ReadError("Premature EOF")

        async def aclose(self):
            pass

    response = MockStreamResponse()

    class MockClient:
        async def aclose(self):
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
