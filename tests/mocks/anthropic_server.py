"""Mock Anthropic API server for bridge end-to-end testing without API keys."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

mock_app = FastAPI(title="mock-anthropic")


def _make_message_response(text: str, model: str = "claude-sonnet-4-20250101") -> dict[str, Any]:
    return {
        "id": "msg_mock_001",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _make_sse_stream(text: str, model: str = "claude-sonnet-4-20250101") -> AsyncGenerator[str, None]:
    """Generate SSE events mimicking Anthropic's streaming format."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_mock_001",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
        {"type": "message_stop"},
    ]

    async def generate() -> AsyncGenerator[str, None]:
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return generate()


# Configurable response mode
_response_mode = {"mode": "normal", "text": None}


def set_response_mode(mode: str, text: str | None = None) -> None:
    _response_mode["mode"] = mode
    _response_mode["text"] = text


@mock_app.post("/v1/messages")
async def messages(request: Request) -> Response | dict[str, Any]:
    body: dict[str, Any] = await request.json()
    is_streaming = body.get("stream", False)

    if _response_mode["mode"] == "malformed_stream" and is_streaming:

        async def malformed() -> AsyncGenerator[bytes, None]:
            yield b"event: message_start\ndata: {invalid json\n\n"
            yield b"event: message_stop\ndata: {}\n\n"

        return StreamingResponse(malformed(), media_type="text/event-stream")

    if _response_mode["mode"] == "tok":
        text = (
            ">>> t:1|usr:test|agt:reply|state:done\n"
            "@thought\n  |> Internal reasoning\n"
            "@msg role:assistant\n  |> This is a Tok response.\n"
        )
    elif _response_mode["text"]:
        text = _response_mode["text"]
    else:
        text = "Hello! This is a mock response from the test server."

    if is_streaming:
        return StreamingResponse(
            _make_sse_stream(text, body.get("model", "claude-sonnet-4-20250101")),
            media_type="text/event-stream",
        )

    return _make_message_response(text, body.get("model", "claude-sonnet-4-20250101"))


@mock_app.get("/{path:path}")
async def catchall(path: str) -> dict[str, str]:
    return {"status": "ok", "path": path}
