import asyncio
import json
import logging
from pathlib import Path
import httpx
from fastapi.testclient import TestClient
import tok.gateway as gateway
from tok.gateway import (
    BridgeSession,
    _buffer_strip_restream,
    _build_tool_use_id_to_context,
    _collect_behavior_signals,
    _record_fallback_once,
    _response_contract,
    create_app,
)
from tok.stats import SavingsTracker
from tok.universal_runtime import PreparedRuntimeRequest


def test_health_endpoint():
    tracker = SavingsTracker(
        savings_file="/tmp/test_health_endpoint_tok_savings.tok",
        ledger_path=Path("/tmp/test_health_endpoint_global_savings.tok"),
    )
    tracker.reset_session_stats()
    app = create_app(BridgeSession(port=9191, tracker=tracker))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "bridge": "tok",
        "port": 9191,
        "mode": "tool-compatible",
        "baseline_only": False,
        "fallback_count": 0,
        "actual_tokens": 0,
        "baseline_tokens": 0,
        "session_tokens_saved": 0,
        "baseline_prompt_tokens": 0,
        "prepared_prompt_tokens": 0,
        "saved_prompt_tokens": 0,
        "session_savings_pct": 0.0,
        "actual_cost_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "cost_saved_usd": 0.0,
        "semantic_drift_count": 0,
        "fail_open_count": 0,
        "non_tok_count": 0,
        "answer_anchor_miss_count": 0,
        "repeat_search_count": 0,
        "repeat_file_read_count": 0,
        "shell_file_read_normalized_count": 0,
        "shell_file_snapshot_captured_count": 0,
        "repeat_target_hot_count": 0,
        "repeat_target_stuck_count": 0,
        "hot_recent_hint_count": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
        "state_resend_full_count": 0,
        "state_resend_delta_count": 0,
        "state_resend_suppressed_count": 0,
        "session_quality": "clean",
        "last_degradation_reason": "",
    }


def test_health_endpoint_reports_baseline_only_and_session_savings(tmp_path):
    tracker = SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )
    tracker.reset_session_stats()
    session = BridgeSession(
        port=9191, memory_dir=tmp_path / ".tok", tracker=tracker
    )
    session.runtime_session._baseline_only = True
    session.tracker.record_call(
        model="claude-sonnet-4",
        actual_input=120,
        actual_output=30,
        cache_read=0,
        cache_write=0,
        input_saved=80,
        output_saved=20,
        behavior_signals={
            "tok_fallback_activated": 2,
            "baseline_only_session": 1,
        },
    )
    app = create_app(session)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_only"] is True
    assert payload["fallback_count"] == 2
    assert payload["session_tokens_saved"] == 100
    assert payload["session_savings_pct"] == 40.0
    assert payload["cost_saved_usd"] > 0
    assert payload["baseline_cost_usd"] > payload["actual_cost_usd"]
    assert payload["session_quality"] == "degraded"
    assert payload["last_degradation_reason"] == "baseline fallback"


def test_clean_system_context_uses_top_level_prompt_compressor(monkeypatch):
    import tok.compression as compression
    from tok.runtime.memory.bridge_memory import (
        BridgeMemoryState,
        clean_system_context,
    )

    captured = {}

    def _fake_compress_user_prompt(prompt: str) -> str:
        captured["prompt"] = prompt
        return "g:repair_bridge|constraints:no_revert"

    monkeypatch.setattr(
        compression, "compress_user_prompt", _fake_compress_user_prompt
    )

    state = BridgeMemoryState(load_global_macros=False)
    result = clean_system_context(state, "Repair bridge immediately")

    assert captured["prompt"] == "Repair bridge immediately"
    assert (
        result
        == "### Optimized Task Context\ng:repair_bridge|constraints:no_revert"
    )


def test_clean_system_context_preserves_cached_list_system_blocks(monkeypatch):
    import tok.compression as compression
    from tok.runtime.memory.bridge_memory import (
        BridgeMemoryState,
        clean_system_context,
    )

    monkeypatch.setattr(
        compression,
        "compress_user_prompt",
        lambda prompt: "g:repair",
    )

    state = BridgeMemoryState(load_global_macros=False)
    result = clean_system_context(
        state,
        [
            {"type": "text", "text": "Prelude"},
            {
                "type": "text",
                "text": "Repair bridge immediately",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "tool_use",
                "id": "sys_tool",
                "name": "noop",
                "input": {},
            },
        ],
    )

    assert isinstance(result, list)
    assert result[0]["text"] == "Prelude"
    assert result[1]["text"] == "### Optimized Task Context\ng:repair"
    assert result[1]["cache_control"]["type"] == "ephemeral"
    assert result[2]["type"] == "tool_use"


def test_record_fallback_once_only_counts_first_event():
    session = BridgeSession()
    request_state = {"fallback_recorded": False}

    _record_fallback_once(session, request_state)
    _record_fallback_once(session, request_state)

    assert session.runtime_session._consecutive_fallback_count == 1
    assert request_state["fallback_recorded"] is True


def test_root_endpoint():
    app = create_app(BridgeSession())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["bridge"] == "tok"


def test_bridge_session_load_memory(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>> g:fix_gateway|t:3\n")

    session = BridgeSession(memory_dir=memory_dir)

    assert session.load_memory() == ">>> t:3|g:fix_gateway"


def test_bridge_session_prefers_structured_memory(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>> g:stale_raw|t:1\n")
    (memory_dir / "bridge_memory.tok").write_text(
        "@mem v:b1 t:2\n@h\n@f turns\n  |> 2|score:3|last:2\n@f goal\n  |> fresh_hot|score:3|last:2\n"
    )

    session = BridgeSession(memory_dir=memory_dir)

    assert (
        session.load_memory(model="claude-sonnet-4") == ">>> t:2|g:fresh_hot"
    )
    assert (
        session.consume_behavior_signals()["cold_start_structured_memory"] == 1
    )


def test_cold_start_request_injects_persisted_memory(tmp_path, monkeypatch):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>> goal:fix_gateway|turns:3\n")

    captured = {}

    async def _fake_send(self, request, stream=False):
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [
                    {"type": "text", "text": "@msg role:assistant\n  |> ok"}
                ],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "Continue."}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert ">>>" in forwarded
    assert "g:fix_gateway" in forwarded


def test_tool_compatible_request_skips_trivial_compressed_history(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>>\n")

    captured = {}

    async def _fake_send(self, request, stream=False):
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "Continue."}],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert "Native tools only. Plain text." in forwarded
    assert "[Tok compressed history]" not in forwarded


def test_bridge_defaults_to_tool_compatible_without_tools(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>>\n")

    captured = {}

    async def _fake_send(self, request, stream=False):
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "Continue."}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert "Native tools only. Plain text." in forwarded


def test_bridge_can_opt_out_of_tool_compatible_via_header(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>>\n")

    captured = {}

    async def _fake_send(self, request, stream=False):
        captured["body"] = request.read()
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test", "x-tok-tool-compatible": "false"},
        json={"model": "claude-sonnet-4", "messages": [], "stream": False},
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert "tool-compatible" not in forwarded
    assert "x-tok-tool-compatible" not in {
        key.lower(): value for key, value in captured["headers"].items()
    }


def test_gateway_retries_with_original_payload_after_tok_prepared_400(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "system": "tok injected system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={"repeat_file_read": 1},
            type_breakdown={"repetitive_cached": 168},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        content = request.read()
        payload = json.loads(content.decode())
        sent_bodies.append(payload)
        if len(sent_bodies) == 1:
            return httpx.Response(
                400, json={"error": {"message": "bad request"}}
            )
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 2
    assert sent_bodies[0]["messages"] == [
        {"role": "user", "content": "compressed by tok"}
    ]
    assert sent_bodies[1] == original_payload


def test_gateway_does_not_retry_when_tok_payload_matches_original(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del request, session, result_cache
        return PreparedRuntimeRequest(
            body=original_payload,
            compressed=True,
            input_saved_tokens=0,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        content = request.read()
        payload = json.loads(content.decode())
        sent_bodies.append(payload)
        return httpx.Response(400, json={"error": {"message": "still bad"}})

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 400
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["model"] == original_payload["model"]
    assert sent_bodies[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert sent_bodies[0]["stream"] == original_payload["stream"]
    assert sent_bodies[0].get("system", "") == ""


def test_gateway_canonicalizes_thinking_blocks_and_logs_preflight_ready(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "internal reasoning",
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            },
                        ],
                    },
                    {
                        "role": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "file contents",
                    },
                    {"role": "user", "content": "Now summarize."},
                ],
                "system": "tok system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=50,
            behavior_signals={"tok_bridge_thinking_block_dropped": 1},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assistant_blocks = sent_bodies[0]["messages"][0]["content"]
    block_types = [b["type"] for b in assistant_blocks]
    assert "thinking" not in block_types
    assert "tool_use" in block_types
    assert "bridge_preflight_ready" in caplog.text
    assert "bridge_preflight_rejected" not in caplog.text


def test_gateway_reverts_when_invalid_block_survives_canonicalization(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "should be removed",
                            },
                            {
                                "type": "custom_unknown_type",
                                "data": "unknown",
                            },
                        ],
                    }
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=10,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert sent_bodies == [original_payload]
    assert "tok_bridge_preflight_rejected" in caplog.text


def test_gateway_canonicalizes_tool_heavy_bridge_body_before_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            }
                        ],
                    },
                    {
                        "role": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "compressed file contents",
                    },
                    {
                        "role": "user",
                        "content": "Summarize the bridge failure.",
                    },
                ],
                "system": "tok injected system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=120,
            behavior_signals={"tok_soft_tool_use_count_high": 1},
            type_breakdown={"repetitive_cached": 480},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert [message["role"] for message in sent_bodies[0]["messages"]] == [
        "assistant",
        "user",
    ]
    assert sent_bodies[0]["messages"][1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "compressed file contents",
        },
        {"type": "text", "text": "Summarize the bridge failure."},
    ]
    assert "bridge_preflight_ready" in caplog.text
    assert "bridge_preflight_rejected" not in caplog.text


def test_gateway_reverts_to_original_when_bridge_preflight_rejects(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": "this is invalid in assistant content",
                            }
                        ],
                    }
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={"repetitive_cached": 168},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert sent_bodies == [original_payload]
    assert "tok_bridge_preflight_rejected" in caplog.text


def test_gateway_reverts_to_original_when_bridge_preflight_rejects_invalid_tool_result_order(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "messages": [
                    {"role": "user", "content": "Inspect the bridge."},
                    {
                        "role": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "compressed log",
                    },
                    {"role": "user", "content": "Summarize the failure."},
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={"semantic_dedup": 168},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert sent_bodies == [original_payload]
    assert "tok_bridge_preflight_rejected" in caplog.text
    assert "user_tool_result_after_text" in caplog.text


def test_gateway_logs_prompt_caching_fingerprint_at_preflight(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "system": [
            {
                "type": "text",
                "text": "System context",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Inspect the bridge",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "tools": [
            {
                "name": "Read",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "stream": False,
    }

    async def _fake_send(self, request, stream=False):
        del self, stream
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "system": "System context",
                "messages": [
                    {"role": "user", "content": "Inspect the bridge"}
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=12,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        json=original_payload,
    )

    assert response.status_code == 200
    assert "anthropic_beta': 'prompt-caching-2024-07-31'" in caplog.text
    assert "'prompt_caching': True" in caplog.text
    assert "'cache_topology_changed': True" in caplog.text
    assert "'system_type_changed'" in caplog.text
    assert "'message_text_cache_control_changed'" in caplog.text
    assert "'removed_text_or_system_cache_control': True" in caplog.text


def test_gateway_reverts_prompt_cached_request_when_cache_topology_changes(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "system": [
            {
                "type": "text",
                "text": "Keep this cached",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "system": "Keep this cached",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=8,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        json=original_payload,
    )

    assert response.status_code == 200
    assert sent_bodies == [original_payload]
    assert "prompt_caching_request_mutated" in caplog.text
    assert "tok_bridge_preflight_rejected" in caplog.text


def test_gateway_allows_prompt_cached_request_when_cache_topology_is_unchanged(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "system": "Original system",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "Read",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "system": "Rewritten system",
                "messages": [{"role": "user", "content": "compressed hello"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=4,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        json=original_payload,
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["system"] == "Rewritten system"
    assert sent_bodies[0]["messages"] == [
        {"role": "user", "content": "compressed hello"}
    ]
    assert "bridge_preflight_ready" in caplog.text
    assert "prompt_caching_request_mutated" not in caplog.text
    assert "'cache_topology_changed': False" in caplog.text


def test_gateway_preflight_reverts_minimal_prompt_cached_tool_shape_before_upstream_retry(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "system": [
            {
                "type": "text",
                "text": "Keep this system cached",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Inspect the bridge",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"path": "src/tok/gateway/__init__.py"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file contents",
                    }
                ],
            },
        ],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "system": "Keep this system cached",
                "messages": [
                    {"role": "user", "content": "Inspect the bridge"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Read",
                                "input": {
                                    "path": "src/tok/gateway/__init__.py"
                                },
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file contents",
                    },
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=16,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        json=original_payload,
    )

    assert response.status_code == 200
    assert sent_bodies == [original_payload]
    assert "prompt_caching_request_mutated" in caplog.text
    assert "Upstream 400 after Tok request preparation" not in caplog.text


def test_gateway_prompt_cached_runtime_request_preserves_system_shape_and_preflight_ready(
    tmp_path, monkeypatch, caplog
):
    import tok.compression as compression

    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_system_text = "Repair the live bridge safely.\n" + (
        "noise\n" * 600
    )
    original_payload = {
        "model": "claude-sonnet-4-6",
        "system": [
            {
                "type": "text",
                "text": original_system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": "Inspect the bridge request path."}
        ],
        "tools": [
            {
                "name": "Read",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    monkeypatch.setattr(
        compression,
        "compress_user_prompt",
        lambda prompt: "g:repair_bridge|constraints:no_revert",
    )

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        json=original_payload,
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert isinstance(sent_bodies[0]["system"], list)
    assert sent_bodies[0]["system"][0]["cache_control"]["type"] == "ephemeral"
    assert "### Optimized Task Context" in sent_bodies[0]["system"][0]["text"]
    assert "bridge_preflight_ready" in caplog.text
    assert "prompt_caching_request_mutated" not in caplog.text


def test_bridge_session_updates_family_mode_from_pressure():
    session = BridgeSession()

    session.update_family_mode(
        "google/gemini-2.0-flash",
        {"repeat_file_read": 1, "repeat_search": 1},
    )

    mode, policy = session.policy_snapshot("google/gemini-2.0-flash")
    assert mode == "balanced"
    assert policy.family.key == "google:gemini"


def test_collect_behavior_signals_detects_repeats_and_workarounds():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "r2",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "s1",
                    "name": "grep",
                    "input": {"query": "create_app"},
                },
                {
                    "type": "tool_use",
                    "id": "s2",
                    "name": "grep",
                    "input": {"query": "create_app"},
                },
                {
                    "type": "tool_use",
                    "id": "b1",
                    "name": "bash",
                    "input": {"command": "python -c 'print(1)' >&2"},
                },
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r1",
            "content": "def route(request):\n    return request\n",
        },
        {
            "role": "tool_result",
            "tool_use_id": "r2",
            "content": "def route(request):\n    return request\n",
        },
        {
            "role": "tool_result",
            "tool_use_id": "s1",
            "content": "src/tok/app.py:10: app = create_app()",
        },
        {
            "role": "tool_result",
            "tool_use_id": "s2",
            "content": "src/tok/app.py:10: app = create_app()",
        },
    ]

    id_to_context = _build_tool_use_id_to_context(messages)
    signals = _collect_behavior_signals(messages, id_to_context)

    assert signals["repeat_file_read"] == 1
    assert signals["repeat_search"] == 1
    assert signals["python_c_workaround"] == 1
    assert signals["stderr_workaround"] == 1
    assert signals.get("reacquisition_cost_tokens", 0) >= 0


def test_response_behavior_signals_detect_non_tok_output():
    from tok.universal_runtime import response_behavior_signals

    assert response_behavior_signals(
        "## hello\nplain markdown", tool_compatible=False
    ) == {"non_tok_response": 1}
    assert (
        response_behavior_signals(
            "@msg role:assistant\n  |> ok", tool_compatible=False
        )
        == {}
    )
    # tool_compatible mode should skip non_tok_response
    assert response_behavior_signals("## markdown", tool_compatible=True) == {}


def test_response_contract_detects_tok_native_success():
    contract = _response_contract(
        ">>> usr:x|agt:y|state:z|t:1\n@msg role:assistant\n  |> ok"
    )

    assert contract.mode == "tok-native"
    assert contract.behavior_signals == {"tok_native_response": 1}
    assert contract.content_blocks == [{"type": "text", "text": "ok"}]


def test_response_contract_detects_non_tok_fail_open_compatibility():
    contract = _response_contract("## heading\nPlain response")

    assert contract.mode == "markdown"
    assert contract.behavior_signals["non_tok_response"] == 1
    assert contract.behavior_signals["fail_open_compat_response"] == 1
    assert contract.content_blocks == [
        {"type": "text", "text": "heading\nPlain response"}
    ]


def test_response_contract_allows_plain_text_in_tool_compatible_mode():
    from tok.gateway import _response_contract_for_mode

    contract = _response_contract_for_mode(
        "Plain response",
        tool_compatible=True,
    )

    assert contract.mode == "tool-compatible"
    assert contract.behavior_signals == {"tool_compatible_response": 1}
    assert contract.content_blocks == [
        {"type": "text", "text": "Plain response"}
    ]


def test_response_contract_detects_malformed_tok_without_native_success():
    contract = _response_contract(">>> g:fix|t:1\n@thought\n  |> hidden only")

    assert contract.behavior_signals["malformed_tok_response"] == 1
    assert "tok_native_response" not in contract.behavior_signals


def test_cold_start_wire_fallback_is_upgraded_to_structured_memory_on_startup(
    tmp_path,
):
    """Bridge startup should ingest raw fallback memory into structured memory immediately."""
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>> goal:debug|turns:5\n")
    # No bridge_memory.tok → structured memory will be empty

    session = BridgeSession(memory_dir=memory_dir)
    result = session.load_memory()

    assert result == ">>> t:5|g:debug"
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_structured_memory", 0) == 1


def test_bridge_session_restores_persisted_fallback_memory_on_startup(
    tmp_path,
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>> g:resume_gateway|t:8\n")

    session = BridgeSession(memory_dir=memory_dir)

    assert (
        session.runtime_session.fallback_memory == ">>> g:resume_gateway|t:8"
    )
    assert session.load_memory() == ">>> t:8|g:resume_gateway"
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_structured_memory", 0) == 1


def test_cold_start_structured_memory_signal_fires_when_bridge_memory_present(
    tmp_path,
):
    """load_memory() should record cold_start_structured_memory when bridge_memory.tok has data."""
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "bridge_memory.tok").write_text(
        "@mem v:b1 t:3\n@h\n@f goal\n  |> fix_tests|score:3|last:3\n"
    )

    session = BridgeSession(memory_dir=memory_dir)
    result = session.load_memory(model="claude-sonnet-4")

    assert "g:fix_tests" in result
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_structured_memory", 0) == 1
    assert signals.get("cold_start_wire_fallback", 0) == 0


def test_fail_open_compat_response_not_emitted_for_tool_compatible_plain_text():
    """Plain text in tool-compatible mode should NOT trigger fail_open_compat_response."""
    from tok.gateway import _response_contract_for_mode

    contract = _response_contract_for_mode(
        "Just a plain answer.", tool_compatible=True
    )

    assert contract.mode == "tool-compatible"
    assert "fail_open_compat_response" not in contract.behavior_signals
    assert "tool_compatible_response" in contract.behavior_signals


def test_fail_open_compat_response_emitted_for_non_tok_in_strict_mode():
    """Non-Tok response in strict mode (tool_compatible=False) must set fail_open_compat_response."""
    from tok.gateway import _response_contract_for_mode

    contract = _response_contract_for_mode(
        "## Some markdown\nNot tok.", tool_compatible=False
    )

    assert contract.behavior_signals.get("fail_open_compat_response", 0) == 1
    assert contract.behavior_signals.get("non_tok_response", 0) == 1


def test_tool_compatible_system_injection_present_when_tools_used(
    tmp_path, monkeypatch
):
    """When the request includes tools, the forwarded system prompt should mention tool-compatible mode."""
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    captured = {}

    async def _fake_send(self, request, stream=False):
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "Continue."}],
            "tools": [
                {
                    "name": "Read",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    body = __import__("json").loads(forwarded)
    system = body.get("system", "")
    assert "Native tools only. Plain text." in system


def test_response_contract_ignores_python_repl_noise_in_tool_compatible_mode():
    from tok.gateway import _response_contract_for_mode

    # A perfectly valid tool-compatible response that includes a Python REPL example
    text = """Certainly! Here is how you use it:
```python
>>> x = [1, 2, 3]
3
```"""
    contract = _response_contract_for_mode(text, tool_compatible=True)

    assert contract.mode == "tool-compatible"
    assert "fail_open_compat_response" not in contract.behavior_signals
    assert "malformed_tok_response" not in contract.behavior_signals
    assert "tok_native_response" not in contract.behavior_signals


def test_response_contract_still_detects_non_tok_in_strict_mode():
    from tok.gateway import _response_contract_for_mode

    # In strict mode, markdown with REPL is still a non-Tok response
    text = """```python\n>>> x = 1\n```"""
    contract = _response_contract_for_mode(text, tool_compatible=False)

    assert contract.behavior_signals.get("non_tok_response") == 1
    assert contract.behavior_signals.get("fail_open_compat_response") == 1


def test_streaming_tool_json_deltas_become_tool_use_blocks():
    class FakeResponse:
        async def aiter_bytes(self):
            payload = "\n\n".join(
                [
                    'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4","usage":{"input_tokens":10,"output_tokens":5}}}',
                    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":">>> usr:test|agt:inspect|state:active|t:0\\n@thought\\n  |> inspecting\\n"}}',
                    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
                    'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}}',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/Users/jfj/Desktop/tok/src/tok/universal_runtime.py\\"}"}}',
                    'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}',
                    'event: message_stop\ndata: {"type":"message_stop"}',
                    "",
                ]
            ).encode()
            yield payload

    class FakeClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    session = BridgeSession()
    client = FakeClient()

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session, client, FakeResponse()
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    output = b"".join(chunks).decode()

    assert '"type": "tool_use"' in output
    assert '"type": "input_json_delta"' in output
    assert "partial_json" in output
    assert "universal_runtime.py" in output
    assert client.closed is True


# ---------------------------------------------------------------------------
# Change 1: inject_system_additions conditionalization tests
# ---------------------------------------------------------------------------


def test_cold_start_no_injection_when_tool_compatible_and_no_memory():
    from tok.compression import inject_system_additions

    body = {"system": "hello"}
    result = inject_system_additions(
        body=body, tok_state=None, tool_compatible=True
    )
    sys = result["system"]
    assert "Native tools only. Plain text." in sys
    assert "[Tok compressed history]" not in sys


def test_tool_compat_with_memory_injects_compat_directive_not_protocol_law():
    from tok.compression import inject_system_additions

    body = {"system": "hello"}
    result = inject_system_additions(
        body=body, tok_state=">>> g:fix|t:2", tool_compatible=True
    )
    sys = result["system"]
    assert "Native tools only. Plain text." in sys
    assert "Always invert multi-line content" not in sys


def test_strict_mode_injection_includes_protocol_law():
    from tok.compression import inject_system_additions

    body = {"system": "hello"}
    # Law threshold is >1: need pressure>=2 to trigger
    result = inject_system_additions(
        body=body, tok_state=None, tool_compatible=False, pressure=2
    )
    sys = result["system"]
    assert "[Tok law]" in sys
    assert "No JSON" in sys


# ---------------------------------------------------------------------------
# Change 2: streaming signal tracking tests
# ---------------------------------------------------------------------------


def test_has_visible_content_block_with_text_block():
    from tok.gateway import _has_visible_content_block

    assert (
        _has_visible_content_block([{"type": "text", "text": "hello"}]) is True
    )
    assert (
        _has_visible_content_block([{"type": "text", "text": "   "}]) is False
    )
    assert _has_visible_content_block([]) is False


def test_has_visible_content_block_with_tool_use_block():
    from tok.gateway import _has_visible_content_block

    assert (
        _has_visible_content_block(
            [{"type": "tool_use", "id": "x", "name": "Read", "input": {}}]
        )
        is True
    )


def test_streaming_tool_only_response_records_tracker():
    from unittest.mock import MagicMock

    class FakeResponse:
        async def aiter_bytes(self):
            payload = "\n\n".join(
                [
                    'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4","usage":{"input_tokens":10,"output_tokens":5}}}',
                    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}}',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/x\\"}"}}',
                    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
                    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
                    'event: message_stop\ndata: {"type":"message_stop"}',
                    "",
                ]
            ).encode()
            yield payload

    class FakeClient:
        async def aclose(self):
            pass

    session = BridgeSession()
    session.tracker.record_call = MagicMock()
    client = FakeClient()

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session, client, FakeResponse()
        ):
            chunks.append(chunk)
        return chunks

    asyncio.run(_collect())
    assert session.tracker.record_call.called


# ---------------------------------------------------------------------------
# Malformed signal coverage
# ---------------------------------------------------------------------------


def test_malformed_tok_hybrid_tool_signals():
    from tok.universal_runtime import malformed_tok_signals

    signals = malformed_tok_signals(
        '@Tool(json={"name": "Read", "path": "/x"})'
    )
    assert signals.get("malformed_tok_hybrid_tool") == 1
    assert signals.get("malformed_tok_response") == 1
