from __future__ import annotations

from typing import Any

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import (
    _detect_recovery_needed,
    _materialize_stream_blocks,
    _parse_sse_stream,
)


def test_parse_sse_stream_extracts_model_and_usage() -> None:
    text = "\n\n".join(
        [
            'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":1}}}',
            'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":2}}',
        ]
    )
    sse_events, accumulated, sse_model, sse_usage, stream_blocks, stream_order = _parse_sse_stream(text)
    assert len(sse_events) == 2
    assert accumulated == []
    assert sse_model == "claude-test"
    assert sse_usage.get("input_tokens") == 1
    assert sse_usage.get("output_tokens") == 2
    assert stream_blocks == {}
    assert stream_order == []


def test_parse_sse_stream_collects_text_deltas() -> None:
    text = "\n\n".join(
        [
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}',
        ]
    )
    _, accumulated, _, _, stream_blocks, stream_order = _parse_sse_stream(text)
    assert "".join(accumulated) == "hi there"
    assert stream_order == [0]
    assert stream_blocks[0]["type"] == "text"
    assert stream_blocks[0]["text"] == "hi there"


def test_materialize_stream_blocks_returns_three_lists(tmp_path) -> None:
    session = BridgeSession(memory_dir=tmp_path / ".tok")
    stream_behavior_signals: dict[str, int] = {}
    stream_blocks: dict[int, dict[str, Any]] = {
        0: {"type": "thinking", "thinking": "x", "signature": ""},
        1: {"type": "text", "text": "start"},
    }
    stream_order = [0, 1]
    tool_blocks, thinking_blocks, text_blocks_from_start = _materialize_stream_blocks(
        stream_blocks,
        stream_order,
        session=session,
        stream_behavior_signals=stream_behavior_signals,
    )
    assert tool_blocks == []
    assert thinking_blocks == [stream_blocks[0]]
    assert text_blocks_from_start == [stream_blocks[1]]


def test_detect_recovery_needed_returns_true_when_no_visible_blocks() -> None:
    assert _detect_recovery_needed(translated_blocks=[], read_error=None) is True


def test_detect_recovery_needed_returns_false_with_tool_use() -> None:
    assert _detect_recovery_needed(translated_blocks=[{"type": "tool_use"}], read_error=None) is False
