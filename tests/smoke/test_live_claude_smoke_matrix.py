"""Automated live-smoke matrix for Claude-through-Tok exercised boundaries."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tok.gateway import BridgeSession, create_app
from tok.runtime.pipeline.request_validation import validate_anthropic_outgoing_bridge_body

_HOST = "127.0.0.1"
_WAIT_RETRIES = 50
_WAIT_SECONDS = 0.1
_REQUEST_TIMEOUT_SECONDS = 30.0
_LONG_PROMPT_CHARS = 24_000
UGLY_PATH_SCENARIOS = (
    "high_fanout_tool_burst",
    "repeated_evidence_loop",
    "final_answer_after_compression",
    "malformed_tool_history",
    "streaming_path_damage",
    "provider_sensitive_shape",
    "baseline_degradation",
    "long_session_retention",
)


class _UpstreamState:
    """Thread-safe state used by synthetic upstream handlers."""

    def __init__(self) -> None:
        self.call_count = 0
        self.lock = asyncio.Lock()
        self.last_request: dict[str, Any] | None = None
        self.requests: list[dict[str, Any]] = []

    async def increment(self, request_body: dict[str, Any]) -> int:
        async with self.lock:
            self.call_count += 1
            self.last_request = request_body
            self.requests.append(request_body)
            return self.call_count

    async def get_count(self) -> int:
        async with self.lock:
            return self.call_count

    async def get_last_request(self) -> dict[str, Any] | None:
        async with self.lock:
            return self.last_request

    async def get_requests(self) -> list[dict[str, Any]]:
        async with self.lock:
            return list(self.requests)


def _file_text(path: str, *, lines: int = 80) -> str:
    return "\n".join(f"{path}:{index}: def marker_{index}(): return {index}" for index in range(lines))


def _tool_use(tool_id: str, name: str, **input_kw: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": input_kw}


def _tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_id, "content": content}


def _request_body(messages: list[dict[str, Any]], *, stream: bool = False, scenario: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "claude-sonnet-4",
        "max_tokens": 512,
        "messages": messages,
        "stream": stream,
    }
    if scenario:
        body["metadata"] = {"ugly_path_scenario": scenario}
    return body


def _high_fanout_messages(*, count: int = 14) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Audit high fanout tool evidence."}]},
        {
            "role": "assistant",
            "content": [
                _tool_use(f"toolu_fanout_{index}", "read_file", path=f"src/tok/fanout_{index}.py")
                for index in range(count)
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(f"toolu_fanout_{index}", _file_text(f"src/tok/fanout_{index}.py", lines=48))
                for index in range(count)
            ],
        },
    ]


def _repeated_evidence_messages(tool_id: str = "toolu_repeat_1") -> list[dict[str, Any]]:
    path = "src/tok/repeated_synthetic.py"
    return [
        {"role": "user", "content": [{"type": "text", "text": "Read repeated evidence."}]},
        {"role": "assistant", "content": [_tool_use(tool_id, "read_file", path=path)]},
        {"role": "user", "content": [_tool_result(tool_id, _file_text(path, lines=120))]},
    ]


def _final_answer_messages() -> list[dict[str, Any]]:
    return [
        *_high_fanout_messages(count=4),
        {"role": "assistant", "content": [{"type": "text", "text": "Ready to answer from evidence."}]},
        {"role": "user", "content": [{"type": "text", "text": "Answer with File= and Verification= only."}]},
    ]


def _malformed_tool_history_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Broken client sent orphaned results."}]},
        {"role": "user", "content": [_tool_result("unknown_toolu_1", "orphaned payload")]},
    ]


def _provider_sensitive_messages() -> list[dict[str, Any]]:
    tool_uses = [_tool_use(f"toolu_sensitive_{index}", "read_file", path=f"file_{index}.py") for index in range(12)]
    return [
        {"role": "user", "content": [{"type": "text", "text": "Inspect provider-sensitive structure."}]},
        {"role": "assistant", "content": [*tool_uses[:6], {"type": "text", "text": "collecting"}, *tool_uses[6:]]},
        {
            "role": "user",
            "content": [_tool_result(f"toolu_sensitive_{index}", f"result {index}") for index in range(12)],
        },
    ]


def _long_retention_messages(turn: int) -> list[dict[str, Any]]:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Seed oldest anchor."}]},
        {"role": "assistant", "content": [_tool_use("toolu_oldest_anchor", "read_file", path="src/tok/gateway.py")]},
        {
            "role": "user",
            "content": [_tool_result("toolu_oldest_anchor", "src/tok/gateway.py:238: async def health()")],
        },
    ]
    for index in range(turn + 1):
        path = f"src/tok/retention_{index}.py"
        messages.extend(
            [
                {"role": "assistant", "content": [{"type": "text", "text": f"Check near neighbor {index}."}]},
                {"role": "user", "content": [{"type": "text", "text": "Keep oldest anchor distinct."}]},
                {
                    "role": "assistant",
                    "content": [_tool_use(f"toolu_retention_{turn}_{index}", "read_file", path=path)],
                },
                {
                    "role": "user",
                    "content": [_tool_result(f"toolu_retention_{turn}_{index}", _file_text(path, lines=24))],
                },
            ]
        )
    return messages


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return sock.getsockname()[1]


def _create_upstream(state: _UpstreamState) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "live-smoke-upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request):
        payload = await request.json()
        await state.increment(payload)

        if bool(payload.get("stream")):
            partial_stream = payload.get("metadata", {}).get("ugly_path_scenario") == "streaming_path_damage"

            async def generate_stream() -> AsyncIterator[bytes]:
                yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
                if partial_stream:
                    return
                yield (
                    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
                    b'"content_block":{"type":"text","text":"stream smoke"}}\n\n'
                )
                yield (b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n')
                yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # Emit tool-use block when caller asks for tool category coverage.
        if payload.get("metadata", {}).get("smoke_category") == "tool-use":
            return JSONResponse(
                content={
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_smoke_1",
                            "name": "view_file",
                            "input": {"path": "README.md"},
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                }
            )

        return JSONResponse(
            content={
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }
        )

    return app


async def _start_server(app: FastAPI, port: int, health_path: str = "/") -> tuple[uvicorn.Server, asyncio.Task]:
    config = uvicorn.Config(app, host=_HOST, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    for _ in range(_WAIT_RETRIES):
        await asyncio.sleep(_WAIT_SECONDS)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://{_HOST}:{port}{health_path}")
                if resp.status_code == 200:
                    return server, task
        except Exception:
            pass

    task.cancel()
    raise RuntimeError(f"server failed to start on port {port}")


async def _stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    with suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)


async def _start_bridge(
    api_base: str,
    bridge_port: int,
    *,
    memory_dir: Path | None = None,
    fail_open: bool = False,
) -> tuple[uvicorn.Server, asyncio.Task]:
    session = BridgeSession(
        port=bridge_port, api_base=api_base, debug=False, fail_open=fail_open, memory_dir=memory_dir
    )
    bridge_app = create_app(session)
    return await _start_server(bridge_app, bridge_port, health_path="/health")


async def _post_bridge(bridge_port: int, body: dict[str, Any], *, session_id: str = "ugly-path") -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"http://{_HOST}:{bridge_port}/v1/messages",
            json=body,
            headers={"x-api-key": "test-api-key", "x-tok-session-id": session_id},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )


async def _bridge_health(bridge_port: int, *, session_id: str = "ugly-path") -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://{_HOST}:{bridge_port}/health",
            headers={"x-api-key": "test-api-key", "x-tok-session-id": session_id},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_live_smoke_tool_use_path() -> None:
    """Pass criterion: tool_use response survives bridge and upstream call count is exactly one."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "list files"}],
            "metadata": {"smoke_category": "tool-use"},
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-api-key"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["content"][0]["type"] == "tool_use"
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_live_smoke_long_context_path() -> None:
    """Pass criterion: long prompt reaches upstream once and bridge returns success without malformed error."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        long_prompt = "A" * _LONG_PROMPT_CHARS
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": long_prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-api-key"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        assert response.status_code == 200
        upstream_payload = await state.get_last_request()
        assert upstream_payload is not None
        assert len(json.dumps(upstream_payload)) >= _LONG_PROMPT_CHARS
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_live_smoke_repeated_call_guard() -> None:
    """Pass criterion: two identical requests produce exactly two upstream calls (no hidden retries)."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "repeat"}],
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            for _ in range(2):
                response = await client.post(
                    f"http://{_HOST}:{bridge_port}/v1/messages",
                    json=request_body,
                    headers={"x-api-key": "test-api-key"},
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                assert response.status_code == 200
        assert await state.get_count() == 2
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_live_smoke_stream_cleanup_early_close() -> None:
    """Pass criterion: early client close does not trigger duplicate upstream execution."""
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(f"http://{_HOST}:{upstream_port}", bridge_port)

    try:
        request_body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "stream then close"}],
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=request_body,
                headers={"x-api-key": "test-api-key"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                assert response.status_code == 200
                async for _chunk in response.aiter_bytes():
                    break
        await asyncio.sleep(0.2)
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


def test_ugly_path_synthetic_matrix_declares_all_supported_scenarios() -> None:
    assert UGLY_PATH_SCENARIOS == (
        "high_fanout_tool_burst",
        "repeated_evidence_loop",
        "final_answer_after_compression",
        "malformed_tool_history",
        "streaming_path_damage",
        "provider_sensitive_shape",
        "baseline_degradation",
        "long_session_retention",
    )


@pytest.mark.asyncio
async def test_ugly_path_synthetic_high_fanout_tool_burst(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        response = await _post_bridge(
            bridge_port,
            _request_body(_high_fanout_messages(), scenario="high_fanout_tool_burst"),
        )
        assert response.status_code == 200
        assert await state.get_count() == 1
        forwarded = await state.get_last_request()
        assert forwarded is not None
        assert validate_anthropic_outgoing_bridge_body(forwarded) == []
        health = await _bridge_health(bridge_port)
        assert health["fallback_count"] == 0
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_repeated_evidence_loop_has_no_hidden_retries(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        for index in range(3):
            response = await _post_bridge(
                bridge_port,
                _request_body(_repeated_evidence_messages(f"toolu_repeat_{index}"), scenario="repeated_evidence_loop"),
            )
            assert response.status_code == 200
        assert await state.get_count() == 3
        requests = await state.get_requests()
        assert all(validate_anthropic_outgoing_bridge_body(request) == [] for request in requests)
        sizes = [len(json.dumps(request)) for request in requests]
        assert max(sizes) <= int(min(sizes) * 1.25) + 512
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_final_answer_after_compression_preserves_response_shape(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        response = await _post_bridge(
            bridge_port,
            _request_body(_final_answer_messages(), scenario="final_answer_after_compression"),
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["content"][0]["type"] == "text"
        assert await state.get_count() == 1
        health = await _bridge_health(bridge_port)
        assert health["baseline_only"] is False
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_malformed_tool_history_blocks_before_upstream(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        response = await _post_bridge(
            bridge_port,
            _request_body(_malformed_tool_history_messages(), scenario="malformed_tool_history"),
        )
        assert response.status_code in {400, 422}
        assert "Traceback" not in response.text
        assert "ValidationError" not in response.text
        assert await state.get_count() == 0
        health = await _bridge_health(bridge_port)
        assert health["fallback_count"] >= 1
        assert health["tool_history_blocked_count"] >= 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_streaming_path_damage_is_exactly_once(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"http://{_HOST}:{bridge_port}/v1/messages",
                json=_request_body(
                    [{"role": "user", "content": "stream partial damage"}],
                    stream=True,
                    scenario="streaming_path_damage",
                ),
                headers={"x-api-key": "test-api-key", "x-tok-session-id": "ugly-path"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                assert response.status_code == 200
                chunks = [chunk async for chunk in response.aiter_bytes()]
        assert any(b"message_start" in chunk for chunk in chunks)
        assert await state.get_count() == 2
        requests = await state.get_requests()
        assert requests[0]["stream"] is True
        assert requests[1]["stream"] is False
        health = await _bridge_health(bridge_port)
        assert health["stream_recovery_attempt_count"] >= 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_provider_sensitive_shape_is_canonicalized(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        response = await _post_bridge(
            bridge_port,
            _request_body(_provider_sensitive_messages(), scenario="provider_sensitive_shape"),
        )
        assert response.status_code == 200
        forwarded = await state.get_last_request()
        assert forwarded is not None
        assert validate_anthropic_outgoing_bridge_body(forwarded) == []
        health = await _bridge_health(bridge_port)
        assert health["fallback_count"] == 0
        assert health["tool_history_pairing_repaired_count"] >= 0
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_baseline_degradation_is_visible_without_upstream(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        malformed = {
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "missing model"}],
            "stream": False,
        }
        response = await _post_bridge(bridge_port, malformed)
        assert response.status_code in {400, 422}
        assert "Traceback" not in response.text
        assert await state.get_count() == 0
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_long_session_retention_uses_stable_session_bucket(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        for turn in range(4):
            response = await _post_bridge(
                bridge_port,
                _request_body(_long_retention_messages(turn), scenario="long_session_retention"),
                session_id="retention-session",
            )
            assert response.status_code == 200
        assert await state.get_count() == 4
        requests = await state.get_requests()
        assert all(validate_anthropic_outgoing_bridge_body(request) == [] for request in requests)
        sizes = [len(json.dumps(request)) for request in requests]
        assert sizes[-1] <= int(sizes[0] * 2.5) + 2048
        health = await _bridge_health(bridge_port, session_id="retention-session")
        assert health["baseline_only"] is False
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)


@pytest.mark.asyncio
async def test_ugly_path_synthetic_health_reports_explicit_session_bucket(tmp_path) -> None:
    state = _UpstreamState()
    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_server, upstream_task = await _start_server(_create_upstream(state), upstream_port)
    bridge_server, bridge_task = await _start_bridge(
        f"http://{_HOST}:{upstream_port}", bridge_port, memory_dir=tmp_path / ".tok"
    )

    try:
        alpha_response = await _post_bridge(
            bridge_port,
            _request_body([{"role": "user", "content": "alpha saves tokens"}]),
            session_id="alpha-health",
        )
        assert alpha_response.status_code == 200
        beta_response = await _post_bridge(
            bridge_port,
            {"max_tokens": 16, "messages": [{"role": "user", "content": "missing model"}], "stream": False},
            session_id="beta-health",
        )
        assert beta_response.status_code in {400, 422}

        alpha_health = await _bridge_health(bridge_port, session_id="alpha-health")
        beta_health = await _bridge_health(bridge_port, session_id="beta-health")

        assert alpha_health["fallback_count"] == 0
        assert beta_health["fallback_count"] >= 1
        assert await state.get_count() == 1
    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)
