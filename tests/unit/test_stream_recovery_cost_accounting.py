"""Regression: stream recovery must preserve original sse_usage cost.

When the upstream stream emits a usable message_start.usage but no visible
content, Tok issues a recovery retry. Both upstream calls have been billed
by the provider, so the tracker must record both. Recording only one leg
silently understates actual_input and overstates savings.
"""

import asyncio
import json

import httpx

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import buffer_strip_restream_impl


def _drive_recovery(tmp_path, monkeypatch) -> BridgeSession:
    session = BridgeSession(memory_dir=tmp_path / ".tok")
    session.tracker.reset_session_stats()
    session.smoothness_tracker.start_turn("turn_1", "task_1")

    class EmptyStreamResponse:
        async def aiter_bytes(self):
            yield (
                b"event: message_start\n"
                b'data: {"type":"message_start","message":'
                b'{"model":"claude-sonnet-4","id":"msg_1",'
                b'"usage":{"input_tokens":1000,"output_tokens":0}}}\n\n'
            )
            yield (
                b"event: message_delta\n"
                b'data: {"delta":{"stop_reason":"end_turn"},'
                b'"type":"message_delta","usage":{"output_tokens":0}}\n\n'
            )

        async def aclose(self) -> None:
            pass

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "content": [{"type": "text", "text": "Recovered answer"}],
                "usage": {"input_tokens": 1000, "output_tokens": 50},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    class FakeClient:
        async def aclose(self) -> None:
            pass

    request_content = json.dumps(
        {
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }
    ).encode()

    async def _drive() -> None:
        async for _ in buffer_strip_restream_impl(
            session=session,
            client=FakeClient(),
            response=EmptyStreamResponse(),
            input_saved_tokens=100,
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=request_content,
            request_state={"fallback_recorded": False},
        ):
            pass

    asyncio.run(_drive())
    return session


def test_stream_recovery_records_both_billed_calls(tmp_path, monkeypatch) -> None:
    session = _drive_recovery(tmp_path, monkeypatch)
    stats = session.tracker.load_stats()
    model_stats = stats["models"].get("claude-sonnet-4", {})
    calls = model_stats.get("calls", 0)
    recorded_input = model_stats.get("actual_input_tokens", 0)

    assert calls >= 2, f"Expected 2 tracker calls (original + recovery), got calls={calls}"
    assert recorded_input >= 2000, (
        f"Stream recovery dropped original sse_usage cost. "
        f"Expected >= 2000 actual_input (1000 stream + 1000 recovery), got {recorded_input}"
    )


def test_stream_recovery_marks_original_empty_call_signal(tmp_path, monkeypatch) -> None:
    session = _drive_recovery(tmp_path, monkeypatch)
    signals = session.tracker.behavior_signals()
    assert signals.get("stream_recovery_original_empty_call", 0) >= 1, (
        f"expected stream_recovery_original_empty_call signal on the original-call record; signals={signals}"
    )
