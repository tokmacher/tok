from __future__ import annotations

import asyncio
import json

import httpx

from tok.gateway import BridgeSession
from tok.gateway._bridge_request_handler import send_with_tok_fail_open_retry
from tok.runtime.smoothness.models import TokMode


def test_fail_open_visible_smoke(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    session.runtime_session._current_tok_mode = TokMode.SMOOTH_MODE

    prepared_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "c"}], "stream": False}
    original_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "o"}], "stream": False}

    sent: list[bytes] = []

    async def _fake_send(_self, request, stream=False):  # type: ignore[no-untyped-def]
        sent.append(request.content)
        if len(sent) == 1:
            return httpx.Response(
                400,
                json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
            )
        return httpx.Response(200, json={"id": "ok", "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}})

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    async def _run():  # type: ignore[no-untyped-def]
        async with httpx.AsyncClient() as client:
            return await send_with_tok_fail_open_retry(
                session,
                client,
                method="POST",
                url="https://example.invalid/v1/messages",
                headers={"x-api-key": "test"},
                content=json.dumps(prepared_body).encode(),
                original_content=json.dumps(original_body).encode(),
                compressed_request=True,
            )

    _response, retried, signals = asyncio.run(_run())
    assert retried is True
    assert signals.get("fail_open_smooth_mode_original_retry") == 1
