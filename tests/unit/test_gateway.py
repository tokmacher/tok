import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
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
from tok.gateway._bridge_preflight import (
    _rewrite_provider_sensitive_large_tool_use_text_interleaving,
)
from tok.gateway._bridge_request_handler import send_with_tok_fail_open_retry
from tok.stats import SavingsTracker
from tok.universal_runtime import PreparedRuntimeRequest
from tok.runtime.pipeline.request_validation import (
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_outgoing_bridge_body,
)
from tok.runtime.memory.bridge_memory import MemoryEntry


def _provider_sensitive_large_tool_batch_messages() -> list[dict[str, Any]]:
    tool_uses = [
        {
            "type": "tool_use",
            "id": f"toolu_batch_{index + 1}",
            "name": "read_file",
            "input": {"path": f"file_{index + 1}.py"},
        }
        for index in range(18)
    ]
    assistant_content = (
        tool_uses[:9]
        + [{"type": "text", "text": "Collecting evidence."}]
        + tool_uses[9:]
    )
    tool_results = [
        {
            "type": "tool_result",
            "tool_use_id": f"toolu_batch_{index + 1}",
            "content": f"result {index + 1}",
        }
        for index in range(18)
    ]
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Inspect concurrency path."}],
        },
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": tool_results},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_tail_1",
                    "name": "grep_search",
                    "input": {"pattern": "stream_recovery"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_tail_1",
                    "content": "tail result",
                }
            ],
        },
    ]


def _interleaved_tool_batch_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Inspect concurrency path."}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_small_1",
                    "name": "read_file",
                    "input": {"path": "file_1.py"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_small_2",
                    "name": "read_file",
                    "input": {"path": "file_2.py"},
                },
                {"type": "text", "text": "Collecting evidence."},
                {
                    "type": "tool_use",
                    "id": "toolu_small_3",
                    "name": "read_file",
                    "input": {"path": "file_3.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_small_1",
                    "content": "result 1",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_small_2",
                    "content": "result 2",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_small_3",
                    "content": "result 3",
                },
            ],
        },
    ]


def test_health_endpoint(monkeypatch):
    monkeypatch.delenv("TOK_MODE", raising=False)
    monkeypatch.delenv("TOK_REQUEST_POLICY", raising=False)

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
        "request_policy": "natural_first",
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
        "stream_recovery_attempt_count": 0,
        "stream_recovery_success_text_count": 0,
        "stream_recovery_success_tool_use_count": 0,
        "stream_recovery_fallback_count": 0,
        "stream_recovery_empty_success_count": 0,
        "stream_recovery_read_error_count": 0,
        "tool_history_repaired_count": 0,
        "tool_history_pairing_repaired_count": 0,
        "tool_history_quarantined_count": 0,
        "tool_history_blocked_count": 0,
        "invalid_tool_history_session_reset_count": 0,
        "provider_pairing_disagreement_count": 0,
        "assistant_tool_use_text_interleaving_blocked_count": 0,
        "preflight_block_original_payload_count": 0,
        "preflight_block_rewritten_payload_count": 0,
        "request_policy_natural_first_count": 0,
        "request_policy_tool_compatible_count": 0,
        "request_policy_escalations_count": 0,
        "request_policy_deescalations_count": 0,
        "request_policy_interleaving_downgrades_count": 0,
        "request_policy_reason_stream_recovery_count": 0,
        "request_policy_reason_tool_recovery_count": 0,
        "request_policy_reason_structured_tool_loop_count": 0,
        "request_policy_held_by_recovery_count": 0,
        "session_quality": "clean",
        "last_degradation_reason": "",
    }


def test_bridge_session_defaults_request_policy_to_natural_first(
    monkeypatch,
):
    monkeypatch.delenv("TOK_MODE", raising=False)
    monkeypatch.delenv("TOK_REQUEST_POLICY", raising=False)

    session = BridgeSession()

    assert session.request_policy_default == "natural_first"
    assert session.tool_compatible_default is True


@pytest.mark.parametrize(
    "request_policy_env",
    ["legacy_tool_compatible", "tool-compatible"],
)
def test_bridge_session_accepts_legacy_request_policy_escape_hatch(
    monkeypatch,
    request_policy_env: str,
):
    monkeypatch.delenv("TOK_MODE", raising=False)
    monkeypatch.setenv("TOK_REQUEST_POLICY", request_policy_env)

    session = BridgeSession()

    assert session.request_policy_default == "legacy_tool_compatible"


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


def test_health_endpoint_reports_recovery_and_repair_counters(tmp_path):
    tracker = SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )
    tracker.reset_session_stats()
    session = BridgeSession(
        port=9191, memory_dir=tmp_path / ".tok", tracker=tracker
    )
    session.tracker.record_call(
        model="claude-sonnet-4",
        actual_input=120,
        actual_output=30,
        cache_read=0,
        cache_write=0,
        input_saved=80,
        output_saved=20,
        behavior_signals={
            "stream_recovery_started": 1,
            "stream_recovery_success_text": 1,
            "stream_recovery_success_tool_use": 1,
            "stream_recovery_fallback": 1,
            "stream_recovery_empty_success": 1,
            "stream_recovery_read_error": 1,
            "tok_bridge_tool_history_repaired": 1,
            "tok_bridge_tool_history_pairing_repaired": 1,
            "tok_bridge_invalid_tool_history_quarantined": 1,
            "tok_bridge_invalid_tool_history_blocked": 1,
            "tok_bridge_invalid_tool_history_session_reset": 1,
            "fail_open_retry_upstream_pairing_disagreement": 1,
            "tok_bridge_assistant_tool_use_text_interleaving_blocked": 1,
            "preflight_block_original_payload": 1,
            "preflight_block_rewritten_payload": 1,
            "request_policy_interleaving_downgrades": 1,
            "request_policy_reason_stream_recovery": 1,
            "request_policy_reason_tool_recovery": 1,
            "request_policy_reason_structured_tool_loop": 1,
            "request_policy_held_by_recovery": 1,
        },
    )
    app = create_app(session)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stream_recovery_attempt_count"] == 1
    assert payload["stream_recovery_success_text_count"] == 1
    assert payload["stream_recovery_success_tool_use_count"] == 1
    assert payload["stream_recovery_fallback_count"] == 1
    assert payload["stream_recovery_empty_success_count"] == 1
    assert payload["stream_recovery_read_error_count"] == 1
    assert payload["tool_history_repaired_count"] == 1
    assert payload["tool_history_pairing_repaired_count"] == 1
    assert payload["tool_history_quarantined_count"] == 1
    assert payload["tool_history_blocked_count"] == 1
    assert payload["invalid_tool_history_session_reset_count"] == 1
    assert payload["provider_pairing_disagreement_count"] == 1
    assert payload["assistant_tool_use_text_interleaving_blocked_count"] == 1
    assert payload["preflight_block_original_payload_count"] == 1
    assert payload["preflight_block_rewritten_payload_count"] == 1
    assert payload["request_policy_interleaving_downgrades_count"] == 1
    assert payload["request_policy_reason_stream_recovery_count"] == 1
    assert payload["request_policy_reason_tool_recovery_count"] == 1
    assert payload["request_policy_reason_structured_tool_loop_count"] == 1
    assert payload["request_policy_held_by_recovery_count"] == 1
    assert (
        payload["last_degradation_reason"] == "request-shape incompatibility"
    )


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
                "max_tokens": 8192,
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
            "max_tokens": 8192,
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
                "max_tokens": 8192,
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
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "Continue."}],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert "TOK-NATIVE" in forwarded
    assert "Plain text. Tool calls only. Omit all headers." not in forwarded
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
                "max_tokens": 8192,
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
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "Continue."}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert "TOK-NATIVE" in forwarded
    assert "Plain text. Tool calls only. Omit all headers." not in forwarded


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
                "max_tokens": 8192,
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
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["body"].decode()
    assert "tool-compatible" not in forwarded
    assert "x-tok-tool-compatible" not in {
        key.lower(): value for key, value in captured["headers"].items()
    }


def test_gateway_natural_first_uses_prepared_effective_tool_mode(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    captured: dict[str, Any] = {}

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        captured["request_policy"] = request.request_policy
        captured["request_has_tools"] = request.request_has_tools
        captured["tool_compatible_allowed"] = request.tool_compatible
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "system": "tok injected system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=12,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            request_policy=request.request_policy,
            effective_tool_compatible=False,
            request_policy_escalated=False,
            normalized_tool_events=[],
        )

    def _fake_process_response(
        text, *, model, session, behavior_signals=None, tool_compatible=False
    ):
        del model, session, behavior_signals
        captured["response_text"] = text
        captured["process_tool_compatible"] = tool_compatible
        return MagicMock(
            content_blocks=[{"type": "text", "text": text}],
            output_saved_tokens=0,
            behavior_signals={},
            mode="natural-first",
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(
        gateway._RUNTIME, "process_response", _fake_process_response
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(
        BridgeSession(
            memory_dir=memory_dir,
            request_policy_default="natural_first",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "Read"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert captured["request_policy"] == "natural_first"
    assert captured["request_has_tools"] is True
    assert captured["tool_compatible_allowed"] is True
    assert captured["process_tool_compatible"] is False


def test_gateway_natural_first_uses_recovery_effective_tool_mode(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    captured: dict[str, Any] = {}

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "system": "tok injected system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=12,
            behavior_signals={"request_policy_reason_stream_recovery": 1},
            type_breakdown={},
            mode="balanced",
            request_policy=request.request_policy,
            effective_tool_compatible=True,
            request_policy_escalated=True,
            normalized_tool_events=[],
        )

    def _fake_process_response(
        text, *, model, session, behavior_signals=None, tool_compatible=False
    ):
        del model, session, behavior_signals
        captured["process_tool_compatible"] = tool_compatible
        return MagicMock(
            content_blocks=[{"type": "text", "text": text}],
            output_saved_tokens=0,
            behavior_signals={},
            mode="tool-compatible",
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(
        gateway._RUNTIME, "process_response", _fake_process_response
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(
        BridgeSession(
            memory_dir=memory_dir,
            request_policy_default="natural_first",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "Read"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert captured["process_tool_compatible"] is True


def test_gateway_degrades_interleaved_assistant_tool_use_batch_to_provider_safe(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Inspect concurrency path.",
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
                                "input": {"path": "a.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_2",
                                "name": "Read",
                                "input": {"path": "b.py"},
                            },
                            {
                                "type": "text",
                                "text": "Collecting evidence.",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_3",
                                "name": "Read",
                                "input": {"path": "c.py"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "a",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_2",
                                "content": "b",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_3",
                                "content": "c",
                            },
                        ],
                    },
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            request_policy="natural_first",
            effective_tool_compatible=False,
            request_policy_escalated=False,
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        sent_bodies.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(
        BridgeSession(
            memory_dir=memory_dir,
            request_policy_default="natural_first",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [
                {
                    "role": "user",
                    "content": "Inspect concurrency path.",
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["model"] == "claude-sonnet-4"
    assert sent_bodies[0]["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Inspect concurrency path.",
        }
    ]
    first_assistant = sent_bodies[0]["messages"][1]
    assert first_assistant["role"] == "assistant"
    first_types = [b.get("type") for b in first_assistant["content"]]
    assert first_types == ["text", "tool_use", "tool_use", "tool_use"]
    assert first_assistant["content"][0]["text"] == "Collecting evidence."
    assert sent_bodies[0]["stream"] is False
    assert sent_bodies[0]["system"] == ""


def test_gateway_interleaving_downgrade_updates_health_counters(tmp_path):
    tracker = SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )
    tracker.reset_session_stats()
    session = BridgeSession(
        port=9191, memory_dir=tmp_path / ".tok", tracker=tracker
    )
    session.tracker.record_call(
        model="claude-sonnet-4",
        actual_input=120,
        actual_output=30,
        cache_read=0,
        cache_write=0,
        input_saved=80,
        output_saved=20,
        behavior_signals={
            "tok_bridge_assistant_tool_use_text_interleaving_blocked": 2,
            "preflight_block_rewritten_payload": 2,
            "request_policy_interleaving_downgrades": 2,
        },
    )
    app = create_app(session)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["assistant_tool_use_text_interleaving_blocked_count"] == 2
    assert payload["preflight_block_rewritten_payload_count"] == 2
    assert payload["request_policy_interleaving_downgrades_count"] == 2


def test_gateway_retries_with_original_payload_after_tok_prepared_400(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
        {
            "role": "user",
            "content": [{"type": "text", "text": "compressed by tok"}],
        }
    ]
    # The retry uses the canonicalized (provider-safe) version of the original payload,
    # where string content is converted to text block format for Anthropic API compatibility
    expected_retry_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ],
        "stream": False,
    }
    assert sent_bodies[1] == expected_retry_payload


def test_gateway_does_not_retry_when_tok_payload_matches_original(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict[str, Any]] = []

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
    assert sent_bodies[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert sent_bodies[0]["stream"] == original_payload["stream"]
    assert sent_bodies[0].get("system", "") == ""


def test_gateway_canonicalizes_thinking_blocks_and_logs_preflight_ready(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
    assert "thinking" in block_types
    assert "tool_use" in block_types
    assert "bridge_preflight_ready" in caplog.text
    assert "bridge_preflight_rejected" not in caplog.text


def test_gateway_sanitizes_invalid_historical_tool_ids_before_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu bad/id",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            }
                        ],
                    },
                    {
                        "role": "tool_result",
                        "tool_use_id": "toolu bad/id",
                        "content": "file contents",
                    },
                    {"role": "user", "content": "Now summarize."},
                ],
                "system": "tok system",
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
                "max_tokens": 8192,
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
    assistant_block = sent_bodies[0]["messages"][0]["content"][0]
    user_block = sent_bodies[0]["messages"][1]["content"][0]
    assert assistant_block["id"] == "toolu_bad_id"
    assert user_block["tool_use_id"] == "toolu_bad_id"
    assert "bridge_preflight_repaired_tool_history" in caplog.text
    assert "bridge_preflight_repaired_tool_result_pairing" in caplog.text
    assert "bridge_preflight_ready" in caplog.text
    assert "bridge_preflight_rejected" not in caplog.text


def test_gateway_count_tokens_sanitizes_invalid_historical_tool_ids_before_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict] = []
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu bad/id",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu bad/id",
                        "content": "file contents",
                    }
                ],
            },
        ],
    }

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(200, json={"input_tokens": 42})

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    assert len(sent_bodies) == 1
    assistant_block = sent_bodies[0]["messages"][0]["content"][0]
    user_block = sent_bodies[0]["messages"][1]["content"][0]
    assert assistant_block["id"] == "toolu_bad_id"
    assert user_block["tool_use_id"] == "toolu_bad_id"
    assert "bridge_preflight_repaired_tool_history_count_tokens" in caplog.text
    assert (
        "bridge_preflight_repaired_tool_result_pairing_count_tokens"
        in caplog.text
    )


def test_gateway_count_tokens_blocks_invalid_tool_history_locally(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "",
                        "name": "",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "",
                        "content": "bad payload",
                    }
                ],
            },
        ],
    }

    async def _fail_if_sent(self, request, stream=False):
        del self, request, stream
        raise AssertionError("upstream send should not be called")

    monkeypatch.setattr(httpx.AsyncClient, "send", _fail_if_sent)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "Tok bridge preflight rejected unrepaired tool history before send."
            ),
        },
    }
    assert (
        "bridge_preflight_rejected_blocked_local_count_tokens" in caplog.text
    )


def test_gateway_synthesizes_blank_historical_tool_ids_before_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            },
                            {
                                "type": "tool_use",
                                "name": "view_file",
                                "input": {"path": "src/tok/runtime.py"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "",
                                "content": "first",
                            },
                            {
                                "type": "tool_result",
                                "content": "second",
                            },
                        ],
                    },
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

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
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
    user_blocks = sent_bodies[0]["messages"][1]["content"]
    ids = [block["id"] for block in assistant_blocks]
    assert all(ids)
    assert len(ids) == len(set(ids))
    assert user_blocks[0]["tool_use_id"] == ids[0]
    assert user_blocks[1]["tool_use_id"] == ids[1]
    assert "bridge_preflight_repaired_tool_history" in caplog.text
    assert "bridge_preflight_repaired_tool_result_pairing" in caplog.text
    assert "bridge_preflight_ready" in caplog.text
    assert "bridge_preflight_rejected" not in caplog.text


def test_gateway_reverts_when_invalid_block_survives_canonicalization(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
    expected_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "should be removed"}
                ],
            }
        ],
        "system": "",
        "stream": False,
    }
    assert sent_bodies == [expected_payload]
    assert "bridge_preflight_ready" in caplog.text
    assert "tok_bridge_preflight_rejected" not in caplog.text


def test_gateway_blocks_invalid_tool_history_locally_without_upstream_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "",
                                "name": "",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "",
                                "content": "file contents",
                            }
                        ],
                    },
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

    async def _fail_if_sent(self, request, stream=False):
        del self, request, stream
        raise AssertionError("upstream send should not be called")

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fail_if_sent)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    app = create_app(session)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "Tok bridge preflight rejected unrepaired tool history before send."
            ),
        },
    }
    assert "bridge_preflight_rejected_blocked_local" in caplog.text
    assert "reverted_to_original=True" not in caplog.text
    signals = session.consume_behavior_signals()
    assert signals["tok_bridge_preflight_failed_local"] == 1
    assert signals["tok_bridge_invalid_tool_history_blocked"] == 1
    assert signals["tok_bridge_strict_invalid_tool_use_block"] == 1
    assert signals["tok_bridge_strict_invalid_tool_result_block"] == 1


def test_gateway_quarantines_broken_tool_exchange_and_continues(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "I inspected the repo.",
                            },
                            {
                                "type": "tool_use",
                                "id": "",
                                "name": "",
                                "input": {},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "",
                                "content": "bad payload",
                            },
                            {
                                "type": "text",
                                "text": "Continue from preserved context.",
                            },
                        ],
                    },
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

    async def _fake_send(self, request, stream=False):
        del stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    app = create_app(session)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "please continue"}],
        }
    ]
    assert "bridge_preflight_pairing_degraded_to_provider_safe" in caplog.text
    assert "bridge_preflight_rejected_blocked_local" not in caplog.text


def test_gateway_repeated_invalid_tool_history_recovery_resets_session_state(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "please continue"}],
        "stream": False,
    }

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "",
                                "name": "",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "",
                                "content": "bad payload",
                            }
                        ],
                    },
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

    async def _fail_if_sent(self, request, stream=False):
        del self, request, stream
        raise AssertionError("upstream send should not be called")

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fail_if_sent)

    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    session.runtime_session._last_tool_compatible_state = "stale"
    session.runtime_session._last_tool_compatible_state_fields = {
        "turns": ["bad"]
    }
    session.runtime_session._observed_tool_result_ids["toolu_bad"] = None
    session.runtime_session.bridge_memory.hot["turns"] = [
        MemoryEntry(value="bad")
    ]
    session.runtime_session.bridge_memory.rolling_cmds = [
        MemoryEntry(value="cat foo")
    ]

    app = create_app(session)
    client = TestClient(app)

    first = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )
    assert first.status_code == 400
    session.consume_behavior_signals()

    second = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )
    assert second.status_code == 400
    signals = session.consume_behavior_signals()
    # After repeated invalid tool history (empty_messages due to stripped invalid blocks),
    # the session state should be reset
    assert (
        signals.get("tok_bridge_invalid_tool_history_session_reset", 0) >= 1
        or signals.get("tok_bridge_preflight_rejected_blocked_local", 0) >= 1
    )
    # Session state should be cleared after repeated failures
    assert session.runtime_session._last_tool_compatible_state == ""
    assert session.runtime_session._last_tool_compatible_state_fields == {}
    assert session.runtime_session._observed_tool_result_ids == {}
    assert "turns" not in session.runtime_session.bridge_memory.hot
    assert session.runtime_session.bridge_memory.rolling_cmds == []


def test_gateway_developer_smoke_surfaces_recovery_and_repair_outcomes(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    class FakeStreamResponse:
        async def aiter_bytes(self):
            if False:
                yield b""
            raise httpx.ReadError("boom")

    class FakeClient:
        async def aclose(self):
            pass

    async def _fake_send(self, request, stream=False):
        del self, stream
        if "example.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 8192,
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "",
                            "name": "Read",
                            "input": {"file_path": "/tmp/example.py"},
                        }
                    ],
                    "usage": {"input_tokens": 8, "output_tokens": 3},
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "",
                                "name": "Read",
                                "input": {"file_path": "/tmp/example.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "",
                                "content": "example contents",
                            }
                        ],
                    },
                    {"role": "user", "content": "Summarize the result."},
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=24,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    caplog.set_level(logging.INFO, logger="tok.gateway")

    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    session.tracker.record_call = MagicMock()

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            FakeClient(),
            FakeStreamResponse(),
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=json.dumps({"stream": True}).encode(),
            request_state={"fallback_recorded": False},
        ):
            chunks.append(chunk)
        return chunks

    output = b"".join(asyncio.run(_collect())).decode()
    assert '"id": "toolu_recovery_1"' in output

    app = create_app(session)
    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "Continue."}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert "stream_recovery_retry_started" in caplog.text
    assert "stream_recovery_succeeded_tool_use" in caplog.text
    assert "bridge_preflight_repaired_tool_history" in caplog.text
    assert "bridge_preflight_repaired_tool_result_pairing" in caplog.text
    assert "bridge_preflight_rejected_blocked_local" not in caplog.text


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
        "user",
    ]
    assert sent_bodies[0]["messages"][1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "compressed file contents",
        }
    ]
    assert sent_bodies[0]["messages"][2]["content"] == [
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
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
    # The gateway reverts to the canonicalized original payload (string content converted to text blocks)
    expected_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ],
        "stream": False,
    }
    assert sent_bodies == [expected_payload]
    assert "tok_bridge_preflight_rejected" in caplog.text


def test_gateway_reverts_to_original_when_bridge_preflight_rejects_invalid_tool_result_order(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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

    # Invalid prepared ordering degrades to a provider-safe original payload.
    assert response.status_code == 200
    expected_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ],
        "stream": False,
    }
    assert sent_bodies == [expected_payload]
    assert "bridge_preflight_pairing_degraded_to_provider_safe" in caplog.text
    assert "tok_bridge_preflight_rejected_blocked_local" not in caplog.text


def test_gateway_logs_prompt_caching_fingerprint_at_preflight(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
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
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
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
        "max_tokens": 8192,
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
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
        "max_tokens": 8192,
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
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
        {
            "role": "user",
            "content": [{"type": "text", "text": "compressed hello"}],
        }
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
        "max_tokens": 8192,
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
                "max_tokens": 8192,
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
                "max_tokens": 8192,
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
    assert mode == "tok-universal"
    assert policy.family.key == "universal:universal"  # type: ignore[union-attr]


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
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "bash",
                    "input": {
                        "command": "uv run pytest tests/unit/test_gateway.py -q"
                    },
                },
                {
                    "type": "tool_use",
                    "id": "c2",
                    "name": "bash",
                    "input": {
                        "command": "uv run pytest tests/unit/test_gateway.py -q"
                    },
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
        {
            "role": "tool_result",
            "tool_use_id": "c1",
            "content": "1 passed in 0.10s",
        },
        {
            "role": "tool_result",
            "tool_use_id": "c2",
            "content": "1 passed in 0.10s",
        },
    ]

    id_to_context = _build_tool_use_id_to_context(messages)
    signals = _collect_behavior_signals(messages, id_to_context)

    assert signals["repeat_file_read"] == 1
    assert signals["repeat_search"] == 1
    assert signals["python_c_workaround"] == 1
    assert signals["stderr_workaround"] == 1
    assert signals["repeat_command"] == 1
    assert signals["repeat_command_stable_no_change"] == 1
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
                "max_tokens": 8192,
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
            "max_tokens": 8192,
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
    assert "TOK-NATIVE" in system


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
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"src/tok/universal_runtime.py\\"}"}}',
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
    assert "Plain text. Tool calls only. Omit all headers." in sys
    assert "[Tok compressed history]" not in sys


def test_tool_compat_with_memory_injects_compat_directive_not_protocol_law():
    from tok.compression import inject_system_additions

    body = {"system": "hello"}
    result = inject_system_additions(
        body=body, tok_state=">>> g:fix|t:2", tool_compatible=True
    )
    sys = result["system"]
    assert "Plain text. Tool calls only. Omit all headers." in sys
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


def test_streaming_empty_success_recovers_via_non_stream_text(
    monkeypatch, caplog
):
    class FakeResponse:
        async def aiter_bytes(self):
            if False:
                yield b""
            raise httpx.ReadError("boom")

    class FakeClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    async def _fake_send(self, request, stream=False):
        assert stream is False
        payload = json.loads(request.content.decode())
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "content": [{"type": "text", "text": "Recovered answer"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    session = BridgeSession()
    session.tracker.record_call = MagicMock()
    client = FakeClient()
    caplog.set_level(logging.INFO, logger="tok.gateway")

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            client,
            FakeResponse(),
            input_saved_tokens=12,
            type_breakdown={"semantic_dedup": 48},
            behavior_signals={},
            prompt_metrics={"saved_prompt_tokens": 12},
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={"x-test": "1"},
            request_content=json.dumps({"stream": True}).encode(),
            request_state={"fallback_recorded": False},
        ):
            chunks.append(chunk)
        return chunks

    output = b"".join(asyncio.run(_collect())).decode()

    assert "Recovered answer" in output
    session.tracker.record_call.assert_called_once()
    signals = session.tracker.record_call.call_args.kwargs["behavior_signals"]
    assert signals["stream_buffer_read_error"] >= 1
    assert signals["stream_empty_after_success"] >= 1
    assert signals["stream_recovery_read_error"] >= 1
    assert signals["stream_recovery_started"] >= 1
    assert signals["stream_recovery_retry"] >= 1
    assert signals["stream_recovery_success_text"] >= 1
    assert signals["tool_compatible_response"] == 1
    assert "stream_recovery_retry_started" in caplog.text
    assert "stream_recovery_succeeded_text" in caplog.text
    assert client.closed is True


def test_streaming_empty_success_recovers_via_non_stream_tool_use(
    monkeypatch, caplog
):
    class FakeResponse:
        async def aiter_bytes(self):
            if False:
                yield b""
            raise httpx.ReadError("boom")

    class FakeClient:
        async def aclose(self):
            pass

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "content": [
                    {
                        "type": "tool_use",
                        "id": "",
                        "name": "Read",
                        "input": {"file_path": "/tmp/example.py"},
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    session = BridgeSession()
    session.tracker.record_call = MagicMock()
    caplog.set_level(logging.INFO, logger="tok.gateway")

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            FakeClient(),
            FakeResponse(),
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=json.dumps({"stream": True}).encode(),
            request_state={"fallback_recorded": False},
        ):
            chunks.append(chunk)
        return chunks

    output = b"".join(asyncio.run(_collect())).decode()

    assert '"type": "tool_use"' in output
    assert '"id": ""' not in output
    assert '"id": "toolu_recovery_1"' in output
    assert "input_json_delta" in output
    assert "/tmp/example.py" in output
    session.tracker.record_call.assert_called_once()
    signals = session.tracker.record_call.call_args.kwargs["behavior_signals"]
    assert signals["stream_recovery_read_error"] >= 1
    assert signals["stream_recovery_started"] >= 1
    assert signals["stream_recovery_retry"] >= 1
    assert signals["stream_recovery_success_tool_use"] >= 1
    assert "stream_recovery_retry_started" in caplog.text
    assert "stream_recovery_succeeded_tool_use" in caplog.text
    assert "tool_compatible_response" not in signals


def test_streaming_empty_success_without_read_error_records_empty_counter(
    monkeypatch, caplog
):
    class FakeResponse:
        async def aiter_bytes(self):
            yield b""

    class FakeClient:
        async def aclose(self):
            pass

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
    session = BridgeSession()
    session.tracker.record_call = MagicMock()
    caplog.set_level(logging.INFO, logger="tok.gateway")

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            FakeClient(),
            FakeResponse(),
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=json.dumps({"stream": True}).encode(),
            request_state={"fallback_recorded": False},
        ):
            chunks.append(chunk)
        return chunks

    output = b"".join(asyncio.run(_collect())).decode()

    assert "Recovered answer" in output
    session.tracker.record_call.assert_called_once()
    signals = session.tracker.record_call.call_args.kwargs["behavior_signals"]
    assert signals["stream_empty_after_success"] >= 1
    assert signals["stream_recovery_empty_success"] >= 1
    assert "stream_recovery_read_error" not in signals
    assert "stream_recovery_retry_started" in caplog.text
    assert "stream_recovery_succeeded_text" in caplog.text


def test_streaming_empty_success_records_fallback_without_tool_compatible_signal(
    monkeypatch, caplog
):
    class FakeResponse:
        async def aiter_bytes(self):
            if False:
                yield b""
            raise httpx.ReadError("boom")

    class FakeClient:
        async def aclose(self):
            pass

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "content": [],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    session = BridgeSession()
    session.tracker.record_call = MagicMock()
    request_state = {"fallback_recorded": False}
    caplog.set_level(logging.INFO, logger="tok.gateway")

    async def _collect():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            FakeClient(),
            FakeResponse(),
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=json.dumps({"stream": True}).encode(),
            request_state=request_state,
        ):
            chunks.append(chunk)
        return chunks

    asyncio.run(_collect())

    assert request_state["fallback_recorded"] is True
    session.tracker.record_call.assert_called_once()
    signals = session.tracker.record_call.call_args.kwargs["behavior_signals"]
    assert signals["stream_buffer_read_error"] >= 1
    assert signals["stream_empty_after_success"] >= 1
    assert signals["stream_recovery_read_error"] >= 1
    assert signals["stream_recovery_started"] >= 1
    assert signals["stream_recovery_retry"] >= 1
    assert signals["stream_recovery_fallback"] >= 1
    assert "stream_recovery_retry_started" in caplog.text
    assert "stream_recovery_fallback" in caplog.text
    assert "tool_compatible_response" not in signals
    assert session.tracker.record_call.called


def test_streaming_empty_success_tool_use_loop_breaker_falls_back(monkeypatch):
    class FakeResponse:
        async def aiter_bytes(self):
            if False:
                yield b""
            raise httpx.ReadError("boom")

    class FakeClient:
        async def aclose(self):
            pass

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "content": [
                    {
                        "type": "tool_use",
                        "id": "",
                        "name": "Bash",
                        "input": {
                            "command": "uv run pytest tests/unit/test_gateway.py -q"
                        },
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    monkeypatch.setenv("TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT", "2")
    session = BridgeSession()
    session.tracker.record_call = MagicMock()
    request_state = {"fallback_recorded": False}

    async def _collect_once():
        chunks = []
        async for chunk in _buffer_strip_restream(
            session,
            FakeClient(),
            FakeResponse(),
            tool_compatible=True,
            request_method="POST",
            request_url="https://example.com/v1/messages",
            request_headers={},
            request_content=json.dumps({"stream": True}).encode(),
            request_state=request_state,
        ):
            chunks.append(chunk)
        return b"".join(chunks).decode()

    first = asyncio.run(_collect_once())
    second = asyncio.run(_collect_once())

    assert '"type": "tool_use"' in first
    assert '"type": "tool_use"' not in second
    assert request_state["fallback_recorded"] is True
    assert session.tracker.record_call.call_count == 2
    signals = session.tracker.record_call.call_args.kwargs["behavior_signals"]
    assert signals["stream_recovery_loop_broken"] == 1
    assert signals["stream_recovery_fallback"] == 1


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


# ---------------------------------------------------------------------------
# M2: Non-streaming processing error fail-open
# ---------------------------------------------------------------------------


def test_non_streaming_processing_error_fail_open_passes_raw_content(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=10,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    def _fake_process_response(*args, **kwargs):
        raise RuntimeError("processing failure")

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "content": [{"type": "text", "text": "raw upstream response"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(
        gateway._RUNTIME, "process_response", _fake_process_response
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
    resp_json = response.json()
    assert resp_json["content"] == [
        {"type": "text", "text": "raw upstream response"}
    ]


def test_non_streaming_processing_error_fail_open_records_usage(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=10,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    def _fake_process_response(*args, **kwargs):
        raise RuntimeError("processing failure")

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "content": [{"type": "text", "text": "raw response"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(
        gateway._RUNTIME, "process_response", _fake_process_response
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    app = create_app(session)
    client = TestClient(app)

    client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    summary = session.tracker.session_summary()
    assert summary is not None
    assert summary["actual_tokens"] > 0


def test_non_streaming_processing_error_fail_closed_propagates(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=10,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    def _fake_process_response(*args, **kwargs):
        raise RuntimeError("processing failure")

    async def _fake_send(self, request, stream=False):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "content": [{"type": "text", "text": "raw response"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(
        gateway._RUNTIME, "process_response", _fake_process_response
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=False))
    client = TestClient(app, raise_server_exceptions=True)

    with pytest.raises(RuntimeError):
        client.post(
            "/v1/messages",
            headers={"x-api-key": "test"},
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )


# ---------------------------------------------------------------------------
# M3: Streaming fail-open retry on HTTP 400
# ---------------------------------------------------------------------------


def test_gateway_retries_streaming_with_original_after_tok_400(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    sent_bodies: list[dict[str, Any]] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "system": "tok injected system",
                "stream": True,
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
                400,
                json={"error": {"message": "bad request"}},
            )
        sse_data = (
            "event: message_start\ndata: "
            + json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-sonnet-4",
                        "max_tokens": 8192,
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                }
            )
            + "\n\n"
            + "event: content_block_start\ndata: "
            + json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            + "\n\n"
            + "event: content_block_delta\ndata: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                }
            )
            + "\n\n"
            + "event: content_block_stop\ndata: "
            + json.dumps({"type": "content_block_stop", "index": 0})
            + "\n\n"
            + "event: message_delta\ndata: "
            + json.dumps(
                {
                    "type": "message_delta",
                    "usage": {"output_tokens": 1},
                }
            )
            + "\n\n"
            + "event: message_stop\ndata: "
            + json.dumps({"type": "message_stop"})
            + "\n\n"
        )
        resp = httpx.Response(
            200,
            content=sse_data.encode(),
            headers={"content-type": "text/event-stream"},
        )
        return resp

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
        {
            "role": "user",
            "content": [{"type": "text", "text": "compressed by tok"}],
        }
    ]
    # The retry uses the canonicalized version where string content is converted to text blocks
    assert sent_bodies[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_gateway_streaming_upstream_400_returns_error_without_stream_recovery(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "system": "tok injected system",
                "stream": True,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        if len(sent_bodies) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "prepared request rejected"}},
            )
        return httpx.Response(
            400,
            json={"error": {"message": "provider-safe retry still rejected"}},
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
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["error"]["message"]
        == "provider-safe retry still rejected"
    )
    assert len(sent_bodies) == 2
    assert "stream_recovery_retry_started" not in caplog.text
    assert "stream_recovery_fallback" not in caplog.text


def test_gateway_retries_upstream_429_then_succeeds(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=1,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        sent_bodies.append(json.loads(request.read().decode()))
        if len(sent_bodies) == 1:
            return httpx.Response(
                429, json={"error": {"message": "slow down"}}
            )
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    monkeypatch.setattr("tok.gateway._app_factory.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(
        "tok.gateway._app_factory.random.uniform", lambda a, b: 1.0
    )
    caplog.set_level(logging.INFO, logger="tok.gateway")

    session = BridgeSession(
        memory_dir=memory_dir,
        fail_open=True,
        rate_limit_retry_max_attempts=2,  # type: ignore[call-arg]
        rate_limit_backoff_base_ms=150,  # type: ignore[call-arg]
        rate_limit_backoff_cap_ms=1000,  # type: ignore[call-arg]
    )
    app = create_app(session)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 2
    assert sleep_calls == [0.15]
    assert "rate_limit_retry_attempt" in caplog.text


def test_gateway_429_retry_honors_retry_after_floor(tmp_path, monkeypatch):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sleep_calls: list[float] = []
    sent_count = 0

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=1,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        nonlocal sent_count
        del self, request, stream
        sent_count += 1
        if sent_count == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
                headers={"Retry-After": "2"},
            )
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    monkeypatch.setattr("tok.gateway._app_factory.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(
        "tok.gateway._app_factory.random.uniform", lambda a, b: 1.0
    )

    app = create_app(
        BridgeSession(
            memory_dir=memory_dir,
            rate_limit_retry_max_attempts=2,  # type: ignore[call-arg]
            rate_limit_backoff_base_ms=150,  # type: ignore[call-arg]
            rate_limit_backoff_cap_ms=1000,  # type: ignore[call-arg]
        )
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert sent_count == 2
    assert sleep_calls == [2.0]


def test_gateway_persistent_429_retries_then_exhausts(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_count = 0

    async def _fake_sleep(_delay: float) -> None:
        return None

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=1,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        nonlocal sent_count
        del self, request, stream
        sent_count += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    monkeypatch.setattr("tok.gateway._app_factory.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(
        "tok.gateway._app_factory.random.uniform", lambda a, b: 1.0
    )
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(
        BridgeSession(
            memory_dir=memory_dir,
            rate_limit_retry_max_attempts=2,  # type: ignore[call-arg]
        )
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 429
    assert sent_count == 3
    assert "rate_limit_retry_exhausted" in caplog.text


def test_gateway_local_throttle_blocks_follow_up_request(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_count = 0

    async def _fake_sleep(_delay: float) -> None:
        return None

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=1,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        nonlocal sent_count
        del self, request, stream
        sent_count += 1
        return httpx.Response(
            429,
            json={"error": {"message": "slow down"}},
            headers={"Retry-After": "5"},
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    monkeypatch.setattr("tok.gateway._app_factory.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(
        "tok.gateway._app_factory.random.uniform", lambda a, b: 1.0
    )

    session = BridgeSession(
        memory_dir=memory_dir,
        rate_limit_retry_max_attempts=0,  # type: ignore[call-arg]
        rate_limit_throttle_threshold=1,  # type: ignore[call-arg]
        rate_limit_throttle_cooldown_sec=20,  # type: ignore[call-arg]
        rate_limit_throttle_window_sec=30,  # type: ignore[call-arg]
    )
    app = create_app(session)
    client = TestClient(app)

    first = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    second = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert first.status_code == 429
    assert second.status_code == 429
    assert second.json()["error"]["type"] == "rate_limit_error"
    assert "Retry-After" in second.headers
    assert sent_count == 1


def test_gateway_local_throttle_expiry_allows_upstream_again(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_count = 0

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=1,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        nonlocal sent_count
        del self, request, stream
        sent_count += 1
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    session = BridgeSession(memory_dir=memory_dir)
    session._rate_limit_throttle_until = time.time() + 30  # type: ignore[attr-defined]
    app = create_app(session)
    client = TestClient(app)

    blocked = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert blocked.status_code == 429
    assert sent_count == 0

    session._rate_limit_throttle_until = time.time() - 1  # type: ignore[attr-defined]
    allowed = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert allowed.status_code == 200
    assert sent_count == 1


# ---------------------------------------------------------------------------
# M5: fail_open=False gating
# ---------------------------------------------------------------------------


def test_gateway_fail_open_false_does_not_retry_on_400(tmp_path, monkeypatch):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed"}],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        content = request.read()
        payload = json.loads(content.decode())
        sent_bodies.append(payload)
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=False))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
    assert len(sent_bodies) == 1


def test_gateway_degrades_to_provider_safe_body_on_prepared_pairing_failure(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {"role": "user", "content": "Inspect the bridge."},
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
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "file contents",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Continue."}],
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
                "max_tokens": 8192,
                "messages": [
                    {"role": "user", "content": "Inspect the bridge."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_2",
                                "name": "grep",
                                "input": {"pattern": "TODO"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": "file contents",
                            }
                        ],
                    },
                ],
                "system": "tok system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
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
    assert sent_bodies[0]["messages"] != [
        {"role": "user", "content": "Inspect the bridge."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "grep",
                    "input": {"pattern": "TODO"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "file contents",
                }
            ],
        },
    ]
    assert sent_bodies[0]["messages"][1]["content"][0]["id"] == "tool_1"
    assert "bridge_preflight_pairing_degraded_to_provider_safe" in caplog.text


def test_gateway_reorders_prepared_out_of_order_tool_results_before_send(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {"role": "user", "content": "Inspect the bridge."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_2",
                                "name": "grep",
                                "input": {"pattern": "TODO"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_2",
                                "content": "grep output",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": "file output",
                            },
                        ],
                    },
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
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
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "file output",
        },
        {
            "type": "tool_result",
            "tool_use_id": "tool_2",
            "content": "grep output",
        },
    ]


def test_gateway_blocks_pairing_failure_when_provider_safe_body_is_still_invalid(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
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
                "role": "user",
                "content": [{"type": "text", "text": "Continue."}],
            },
        ],
        "stream": False,
    }

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "view_file",
                                "input": {"path": "src/tok/gateway.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_2",
                                "name": "grep",
                                "input": {"pattern": "TODO"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Continue."}],
                    },
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fail_if_sent(self, request, stream=False):
        del self, request, stream
        raise AssertionError("upstream send should not be called")

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fail_if_sent)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=True))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json=original_payload,
    )

    assert response.status_code == 400
    assert "bridge_preflight_rejected_blocked_local" in caplog.text
    assert (
        "bridge_preflight_pairing_degraded_to_provider_safe" not in caplog.text
    )


def test_gateway_fail_open_retry_uses_provider_safe_body_after_tool_history_repair(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    original_payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu bad/id",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu bad/id",
                        "content": "file contents",
                    },
                    {"type": "text", "text": "Continue."},
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
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "compressed by tok"}],
                "system": "tok injected system",
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        if len(sent_bodies) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "bad request"}},
            )
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
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
        {
            "role": "user",
            "content": [{"type": "text", "text": "compressed by tok"}],
        }
    ]
    retry_text = json.dumps(sent_bodies[1], sort_keys=True)
    assert "toolu bad/id" not in retry_text
    assert "toolu_bad_id" in retry_text


def test_provider_sensitive_large_file_read_burst_rewrite_preserves_pairing():
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": _provider_sensitive_large_tool_batch_messages(),
    }

    rewritten, changed, signals = (
        _rewrite_provider_sensitive_large_tool_use_text_interleaving(body)
    )

    assert changed is True
    assert signals["tok_bridge_large_file_read_burst_rewritten"] == 1
    assert validate_anthropic_outgoing_bridge_body(rewritten) == []

    original_timeline = summarize_bridge_pairing(body["messages"])
    rewritten_timeline = summarize_bridge_pairing(rewritten["messages"])

    def _flatten(entries: list[dict[str, Any]], key: str) -> list[str]:
        values: list[str] = []
        for entry in entries:
            values.extend([str(item) for item in entry.get(key, [])])
        return values

    assert _flatten(rewritten_timeline, "tool_use_ids") == _flatten(
        original_timeline, "tool_use_ids"
    )
    assert _flatten(rewritten_timeline, "next_tool_result_ids") == _flatten(
        original_timeline, "next_tool_result_ids"
    )


def test_gateway_rewrites_provider_sensitive_large_tool_batch_before_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(session)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": _provider_sensitive_large_tool_batch_messages(),
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    assert validate_anthropic_outgoing_bridge_body(sent_bodies[0]) == []


def test_gateway_rewrites_small_interleaved_tool_batch_before_send(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(session)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": _interleaved_tool_batch_messages(),
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    sent_messages = sent_bodies[0]["messages"]
    for msg in sent_messages:
        if msg["role"] == "assistant":
            types = [b.get("type") for b in msg["content"]]
            has_tool = "tool_use" in types
            has_text = "text" in types
            if has_tool and has_text:
                tool_indices = [
                    i for i, t in enumerate(types) if t == "tool_use"
                ]
                text_indices = [i for i, t in enumerate(types) if t == "text"]
                assert not any(
                    ti < tool_indices[-1] and ti > tool_indices[0]
                    for ti in text_indices
                ), (
                    f"assistant message has text interleaved between tool_use blocks: {types}"
                )


def test_gateway_rewrites_provider_sensitive_prepared_body_to_safe_segments(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": _provider_sensitive_large_tool_batch_messages(),
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}],
            },
        )

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    app = create_app(session)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test"},
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert len(sent_bodies) == 1
    summary = summarize_message_structure(sent_bodies[0]["messages"])
    summary = cast(dict[str, Any], summary)
    assert validate_anthropic_outgoing_bridge_body(sent_bodies[0]) == []
    assert "canonicalized_changed=True" in caplog.text


def test_gateway_logs_pairing_forensics_when_upstream_reports_pairing_disagreement(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {"role": "user", "content": "Inspect concurrency path."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_a1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_a2",
                                "name": "read_file",
                                "input": {"path": "b.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_a3",
                                "name": "read_file",
                                "input": {"path": "c.py"},
                            },
                            {"type": "text", "text": "Collecting evidence."},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_a1",
                                "content": "A",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_a2",
                                "content": "B",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_a3",
                                "content": "C",
                            },
                        ],
                    },
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        if len(sent_bodies) == 1:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: "
                            "toolu_a1, toolu_a2, toolu_a3. Each `tool_use` block must have a corresponding `tool_result` block in the next message."
                        ),
                    },
                },
            )
        return httpx.Response(
            400,
            json={"error": {"message": "provider-safe retry also rejected"}},
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
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
    assert len(sent_bodies) == 2
    assert "bridge_pairing_forensics" in caplog.text
    assert "fail_open_retry_upstream_pairing_disagreement" in caplog.text
    assert (
        "fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message"
        in caplog.text
    )
    assert "toolu_a1" in caplog.text
    assert "toolu_a2" in caplog.text
    assert "toolu_a3" in caplog.text


def test_gateway_logs_pairing_disagreement_after_user_message_split(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    sent_bodies: list[dict[str, Any]] = []

    def _fake_prepare_request(request, session, *, result_cache=None):
        del session, result_cache
        return PreparedRuntimeRequest(
            body={
                "model": request.model,
                "max_tokens": 8192,
                "messages": [
                    {"role": "user", "content": "Inspect split path."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_a1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_a1",
                                "content": "A",
                            },
                            {"type": "text", "text": "Continue."},
                        ],
                    },
                ],
                "stream": False,
            },
            compressed=True,
            input_saved_tokens=42,
            behavior_signals={},
            type_breakdown={},
            mode="balanced",
            normalized_tool_events=[],
        )

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        if len(sent_bodies) == 1:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: "
                            "toolu_a1. Each `tool_use` block must have a corresponding `tool_result` block in the next message."
                        ),
                    },
                },
            )
        return httpx.Response(
            400,
            json={"error": {"message": "provider-safe retry also rejected"}},
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
        json={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
    assert len(sent_bodies) == 2
    assert sent_bodies[0]["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_a1",
            "content": "A",
        }
    ]
    assert sent_bodies[0]["messages"][3]["content"] == [
        {"type": "text", "text": "Continue."}
    ]
    assert "prepared_mixed_user_tool_result_messages=0" in caplog.text
    assert "prepared_split_boundaries=1" in caplog.text
    assert (
        "fail_open_retry_upstream_pairing_disagreement_after_user_message_split"
        in caplog.text
    )


def test_retry_blocks_provider_sensitive_provider_safe_payload(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    sent_bodies: list[dict[str, Any]] = []
    prepared_body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "compressed"}],
            }
        ],
        "stream": False,
    }
    retry_body = {
        "model": "claude-sonnet-4",
        "messages": _provider_sensitive_large_tool_batch_messages(),
        "stream": False,
    }

    async def _fake_send(self, request, stream=False):
        del self, stream
        payload = json.loads(request.read().decode())
        sent_bodies.append(payload)
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "messages.1: `tool_use` ids were found without "
                        "`tool_result` blocks immediately after: toolu_batch_1"
                    ),
                },
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)
    caplog.set_level(logging.INFO, logger="tok.gateway")

    async def _exercise() -> httpx.Response:
        async with httpx.AsyncClient() as client:
            (
                response,
                retried_without_tok,
                retry_signals,
            ) = await send_with_tok_fail_open_retry(
                session,
                client,
                method="POST",
                url="https://example.invalid/v1/messages",
                headers={"x-api-key": "test"},
                content=json.dumps(prepared_body).encode(),
                original_content=json.dumps(prepared_body).encode(),
                retry_content=json.dumps(retry_body).encode(),
                compressed_request=True,
            )
            assert retried_without_tok is False
            assert retry_signals["fail_open_retry_provider_safe_invalid"] == 1
            assert (
                retry_signals["fail_open_retry_provider_safe_blocked_local"]
                == 1
            )
            return response

    response = asyncio.run(_exercise())

    assert response.status_code == 400
    assert len(sent_bodies) == 1
    assert "fail_open_retry_provider_safe_invalid" in caplog.text
    assert "provider-safe payload failed final local validation" in caplog.text


def test_gateway_fail_open_false_propagates_request_processing_error(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    def _fake_prepare_request(request, session, *, result_cache=None):
        del request, session, result_cache
        raise RuntimeError("request prep failure")

    monkeypatch.setattr(
        gateway._RUNTIME, "prepare_request", _fake_prepare_request
    )

    app = create_app(BridgeSession(memory_dir=memory_dir, fail_open=False))
    client = TestClient(app, raise_server_exceptions=True)

    with pytest.raises(RuntimeError):
        client.post(
            "/v1/messages",
            headers={"x-api-key": "test"},
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
