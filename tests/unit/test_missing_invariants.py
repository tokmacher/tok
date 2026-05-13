"""
Missing invariant tests for Tok 0.2.0.

These tests pin hard invariants that must survive extraction-only refactors.
Each test fails in a controlled way if its invariant is broken.

Packet 0: Missing Invariant Tests
See: docs/plans/0.1.9/packets/packet-00-missing-invariant-tests.md
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from typer.testing import CliRunner

from tok.cli import app
from tok.compression import _compute_semantic_hash
from tok.compression._history_pipeline import compress_tool_results_impl
from tok.gateway import BridgeSession
from tok.gateway._bridge_request_handler import send_with_tok_fail_open_retry
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context
from tok.runtime.pipeline._tool_repeat_detection import _make_cache_key
from tok.runtime.smoothness.models import TokMode
from tok.spec.live_trace import emit_live_trace
from tok.stats import SavingsTracker

runner = CliRunner()


def _tool_use(tool_id: str, tool_name: str, **input_kw: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": input_kw,
            }
        ],
    }


def _tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            }
        ],
    }


class _BridgeMemory:
    turn = 3


class _RuntimeSession:
    bridge_memory = _BridgeMemory()
    _current_tok_mode: TokMode = TokMode.FULL_TOK

    @property
    def current_tok_mode(self) -> TokMode:
        return self._current_tok_mode


class _Session:
    _active_session_key = "test-session"
    _live_trace_instance_id = "test-trace-instance"

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.runtime_session = _RuntimeSession()


# ---------------------------------------------------------------------------
# Dedup / duplicate upstream calls
# ---------------------------------------------------------------------------


class TestDedupFrontierIdempotentOnRetry:
    """Dedup frontier must return identical fingerprints for identical requests."""

    def test_cache_key_idempotent_on_retry(self) -> None:
        tool_name = "read_file"
        context = {"args": {"file_path": "/tmp/test_dedup.py"}}
        key1 = _make_cache_key(tool_name, context)
        key2 = _make_cache_key(tool_name, context)
        assert key1 == key2
        assert len(key1) == 12

    def test_semantic_hash_idempotent_on_retry(self) -> None:
        content = "def hello():\n    return 'world'\n" * 20
        hash1 = _compute_semantic_hash(content)
        hash2 = _compute_semantic_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_cache_key_stable_across_ordering_variants(self) -> None:
        tool_name = "read_file"
        context_a = {"args": {"file_path": "/tmp/a.py", "offset": 0}}
        context_b = {"args": {"offset": 0, "file_path": "/tmp/a.py"}}
        key_a = _make_cache_key(tool_name, context_a)
        key_b = _make_cache_key(tool_name, context_b)
        assert key_a == key_b

    def test_different_args_produce_different_keys(self) -> None:
        tool_name = "read_file"
        context_a = {"args": {"file_path": "/tmp/a.py"}}
        context_b = {"args": {"file_path": "/tmp/b.py"}}
        key_a = _make_cache_key(tool_name, context_a)
        key_b = _make_cache_key(tool_name, context_b)
        assert key_a != key_b


class TestStreamingDedupCachesFirstResponse:
    """First streaming response cached, subsequent retries return cached."""

    def test_second_identical_read_hits_cache(self) -> None:
        file_content = "line " * 100
        messages = [
            _tool_use("t1", "read_file", file_path="/tmp/foo.py"),
            _tool_result("t1", file_content),
            _tool_use("t2", "read_file", file_path="/tmp/foo.py"),
            _tool_result("t2", file_content),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()
        result_cache: dict[str, Any] = {}

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        first_content = compressed[1]["content"][0]["content"]
        assert first_content == file_content

        assert len(result_cache) >= 1, "Result cache must be populated after compression"
        assert breakdown.get("cache_stored", 0) >= 1, "Breakdown must report cache_stored"

        compressed2, breakdown2 = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        assert breakdown2.get("cache_hit", 0) >= 1, "Second pass with populated cache must report cache_hit"
        first_pass2 = compressed2[1]["content"][0]["content"]
        assert first_pass2 == file_content

    def test_cache_populated_after_first_read(self) -> None:
        file_content = "x = 1\n" * 80
        messages = [
            _tool_use("t1", "read_file", file_path="/tmp/cached.py"),
            _tool_result("t1", file_content),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        result_cache: dict[str, Any] = {}

        compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
        )

        assert len(result_cache) >= 1, "Result cache should be populated after first read"

        cache_key = _make_cache_key("read_file", tool_use_id_to_context["t1"])
        assert cache_key in result_cache


# ---------------------------------------------------------------------------
# Fail-open correctness (smooth mode)
# ---------------------------------------------------------------------------


class TestFailOpenRetrySmoothModeOriginalRetry:
    """When SMOOTH_MODE active and 400 received, original payload is retried."""

    def test_smooth_mode_retries_with_original(self, tmp_path, monkeypatch) -> None:
        memory_dir = tmp_path / ".tok"
        memory_dir.mkdir()
        session = BridgeSession(memory_dir=memory_dir, fail_open=True)
        session.runtime_session._current_tok_mode = TokMode.SMOOTH_MODE

        prepared_body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "compressed version"}]}],
            "stream": False,
        }
        original_body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "original version"}]}],
            "stream": False,
        }

        sent_contents: list[bytes] = []
        call_count = 0

        async def _fake_send(_self, request, stream=False):
            nonlocal call_count
            call_count += 1
            sent_contents.append(request.content)
            if call_count == 1:
                return httpx.Response(
                    400,
                    json={
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "bad request"},
                    },
                )
            return httpx.Response(200, json={"id": "msg_ok", "content": []})

        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

        async def _exercise():
            async with httpx.AsyncClient() as client:
                response, retried_without_tok, retry_signals = await send_with_tok_fail_open_retry(
                    session,
                    client,
                    method="POST",
                    url="https://example.invalid/v1/messages",
                    headers={"x-api-key": "test"},
                    content=json.dumps(prepared_body).encode(),
                    original_content=json.dumps(original_body).encode(),
                    compressed_request=True,
                )
                return response, retried_without_tok, retry_signals

        response, retried, signals = asyncio.run(_exercise())

        assert response.status_code == 200
        assert retried is True
        assert signals.get("fail_open_smooth_mode_original_retry") == 1
        assert call_count == 2

        second_payload = json.loads(sent_contents[1])
        assert second_payload == original_body, "Retry should use original payload, not compressed"

    def test_smooth_mode_skips_provider_safe(self, tmp_path, monkeypatch) -> None:
        memory_dir = tmp_path / ".tok"
        memory_dir.mkdir()
        session = BridgeSession(memory_dir=memory_dir, fail_open=True)
        session.runtime_session._current_tok_mode = TokMode.SMOOTH_MODE

        prepared_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "c"}], "stream": False}
        original_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "o"}], "stream": False}
        retry_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "r"}], "stream": False}

        sent_contents: list[bytes] = []

        async def _fake_send(_self, request, stream=False):
            sent_contents.append(request.content)
            if len(sent_contents) == 1:
                return httpx.Response(
                    400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
                )
            return httpx.Response(200, json={"id": "ok"})

        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

        async def _exercise():
            async with httpx.AsyncClient() as client:
                return await send_with_tok_fail_open_retry(
                    session,
                    client,
                    method="POST",
                    url="https://example.invalid/v1/messages",
                    headers={"x-api-key": "test"},
                    content=json.dumps(prepared_body).encode(),
                    original_content=json.dumps(original_body).encode(),
                    retry_content=json.dumps(retry_body).encode(),
                    compressed_request=True,
                )

        _, retried, signals = asyncio.run(_exercise())

        assert retried is True
        assert len(sent_contents) == 2
        second_payload = json.loads(sent_contents[1])
        assert second_payload == original_body, "Smooth mode should skip provider-safe and use original"

    def test_non_smooth_mode_does_not_use_smooth_path(self, tmp_path, monkeypatch) -> None:
        memory_dir = tmp_path / ".tok"
        memory_dir.mkdir()
        session = BridgeSession(memory_dir=memory_dir, fail_open=True)
        session.runtime_session._current_tok_mode = TokMode.FULL_TOK

        prepared_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "c"}], "stream": False}
        original_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "o"}], "stream": False}
        retry_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "r"}], "stream": False}

        sent_contents: list[bytes] = []

        async def _fake_send(_self, request, stream=False):
            sent_contents.append(request.content)
            if len(sent_contents) <= 2:
                return httpx.Response(
                    400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
                )
            return httpx.Response(200, json={"id": "ok"})

        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

        async def _exercise():
            async with httpx.AsyncClient() as client:
                return await send_with_tok_fail_open_retry(
                    session,
                    client,
                    method="POST",
                    url="https://example.invalid/v1/messages",
                    headers={"x-api-key": "test"},
                    content=json.dumps(prepared_body).encode(),
                    original_content=json.dumps(original_body).encode(),
                    retry_content=json.dumps(retry_body).encode(),
                    compressed_request=True,
                )

        _, retried, signals = asyncio.run(_exercise())

        assert "fail_open_smooth_mode_original_retry" not in signals


# ---------------------------------------------------------------------------
# Diagnostics consistency across commands
# ---------------------------------------------------------------------------


class TestStatsDoctorStatusAuditReflectSameBackendState:
    """Stats, doctor, and bridge status must report consistent numbers."""

    def test_stats_and_status_agree_on_tokens(self, tmp_path, monkeypatch) -> None:
        savings_file = str(tmp_path / "tok_savings.tok")
        ledger_path = tmp_path / "global_savings.tok"
        tracker = SavingsTracker(savings_file=savings_file, ledger_path=ledger_path)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=100,
        )
        session_summary = tracker.session_summary()
        assert session_summary is not None

        tokens_saved = int(session_summary["tokens_saved"])
        actual_tokens = int(session_summary["actual_tokens"])
        baseline_tokens = int(session_summary["baseline_tokens"])

        health_payload = {
            "status": "ok",
            "bridge": "tok",
            "port": 9090,
            "mode": "tool-compatible",
            "baseline_only": False,
            "fallback_count": 0,
            "session_tokens_saved": tokens_saved,
            "session_savings_pct": float(session_summary["savings_pct"]),
            "actual_tokens": actual_tokens,
            "baseline_tokens": baseline_tokens,
            "actual_cost_usd": float(session_summary["actual_cost_usd"]),
            "baseline_cost_usd": float(session_summary["baseline_cost_usd"]),
            "cost_saved_usd": float(session_summary["cost_saved_usd"]),
            "session_quality": "clean",
            "last_degradation_reason": "",
            "calls": int(session_summary["calls"]),
        }

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return health_payload

        monkeypatch.setenv("TOK_SAVINGS_FILE", savings_file)
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("httpx.get", lambda *a, **_kw: FakeResponse())

        stats_result = runner.invoke(app, ["stats"])
        assert stats_result.exit_code == 0
        assert str(tokens_saved) in stats_result.output

        status_result = runner.invoke(app, ["bridge", "status"])
        assert status_result.exit_code == 0
        assert str(tokens_saved) in status_result.output

    def test_stats_and_doctor_agree_on_backend_state(self, tmp_path, monkeypatch) -> None:
        savings_file = str(tmp_path / "tok_savings.tok")
        ledger_path = tmp_path / "global_savings.tok"
        tracker = SavingsTracker(savings_file=savings_file, ledger_path=ledger_path)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=2000,
            actual_output=1000,
            cache_read=0,
            cache_write=0,
            input_saved=600,
            output_saved=200,
        )
        session_summary = tracker.session_summary()
        assert session_summary is not None

        tokens_saved = int(session_summary["tokens_saved"])
        savings_pct = float(session_summary["savings_pct"])

        memory_dir = tmp_path / ".tok"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "bridge_memory.tok").write_text("{}")
        monkeypatch.setenv("TOK_SAVINGS_FILE", savings_file)
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))

        class FakeHealthResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "tool-compatible",
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_tokens_saved": tokens_saved,
                    "session_savings_pct": savings_pct,
                    "actual_tokens": int(session_summary["actual_tokens"]),
                    "baseline_tokens": int(session_summary["baseline_tokens"]),
                    "actual_cost_usd": float(session_summary["actual_cost_usd"]),
                    "baseline_cost_usd": float(session_summary["baseline_cost_usd"]),
                    "cost_saved_usd": float(session_summary["cost_saved_usd"]),
                    "session_quality": "clean",
                    "last_degradation_reason": "",
                    "calls": int(session_summary["calls"]),
                }

        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: None)
        monkeypatch.setattr("httpx.get", lambda *a, **_kw: FakeHealthResponse())

        stats_result = runner.invoke(app, ["stats"])
        assert stats_result.exit_code == 0
        assert str(tokens_saved) in stats_result.output

        doctor_result = runner.invoke(app, ["doctor"])
        assert str(tokens_saved) in doctor_result.output


# ---------------------------------------------------------------------------
# Trace/audit privacy
# ---------------------------------------------------------------------------


class TestTraceMetadataContainsNoUpstreamUrls:
    """Trace entries must contain no upstream API URLs."""

    def test_live_trace_blocks_contain_no_upstream_urls(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TOK_TRACE", "1")
        monkeypatch.delenv("TOK_TRACE_FILE", raising=False)
        monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)
        session = _Session(tmp_path)

        emit_live_trace(
            session,
            "request_prepared",
            trace_class="message",
            action="pass_through",
            result="ok",
            expectation="accept_pass_through",
            reason="test",
            direction="request",
            metadata={
                "compressed": True,
                "input_saved_tokens": 300,
                "mode": "tool-compatible",
                "behavior_signals": {"repeat_file_read": 1},
            },
        )

        emit_live_trace(
            session,
            "response_processed",
            trace_class="response",
            action="pass_through",
            result="ok",
            expectation="accept_pass_through",
            reason="test",
            direction="response",
            metadata={
                "output_saved_tokens": 50,
                "behavior_signals": {},
            },
        )

        trace_files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(trace_files) >= 1, "Expected at least one trace file"

        for trace_file in trace_files:
            rendered = trace_file.read_text()
            for line in rendered.splitlines():
                if not line.strip():
                    continue
                block = json.loads(line)
                block_text = json.dumps(block, sort_keys=True)

                session_id = block.get("envelope", {}).get("session_id", "")
                assert not session_id.startswith("http"), f"Session ID should be hashed, not a URL: {session_id}"

                for pattern in ["api.anthropic.com", "api.openai.com", "openrouter.ai"]:
                    assert pattern not in block_text, f"Upstream URL pattern '{pattern}' found in trace block"


class TestAuditLogContainsNoApiCredentials:
    """Audit logs must contain no API keys or credentials."""

    def test_capture_file_redacts_all_secrets(self, tmp_path) -> None:
        session = BridgeSession(capture=True, memory_dir=tmp_path / ".tok")

        session.capture_request(
            {
                "messages": [{"role": "user", "content": "Bearer secret-token-abc123 and sk-live-key-xyz789"}],
                "system": "Authorization: Bearer top-secret-bearer",
                "x-api-key": "sk-top-level-key",
                "anthropic_api_key": "sk-ant-secret123",
            }
        )

        lines = session._capture_file.read_text().splitlines()
        payload = json.loads(lines[-1])
        rendered = json.dumps(payload)

        assert "secret-token-abc123" not in rendered
        assert "sk-live-key-xyz789" not in rendered
        assert "top-secret-bearer" not in rendered
        assert "sk-top-level-key" not in rendered
        assert "sk-ant-secret123" not in rendered
        assert "Bearer <redacted>" in rendered
        assert "sk-<redacted>" in rendered

    def test_live_trace_does_not_leak_api_keys(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TOK_TRACE", "1")
        monkeypatch.delenv("TOK_TRACE_FILE", raising=False)
        monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)
        session = _Session(tmp_path)

        emit_live_trace(
            session,
            "request_prepared",
            trace_class="message",
            action="pass_through",
            result="ok",
            expectation="accept_pass_through",
            reason="test",
            direction="request",
            metadata={
                "model": "claude-sonnet-4",
                "compressed": True,
                "input_saved_tokens": 100,
            },
        )

        trace_files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(trace_files) >= 1

        rendered = trace_files[0].read_text()
        assert "sk-" not in rendered or "sk-" in rendered.split('"session_id"')[0] == rendered
        for line in rendered.splitlines():
            if not line.strip():
                continue
            block = json.loads(line)
            block_text = json.dumps(block, sort_keys=True)
            assert "sk-ant-" not in block_text
            assert "Bearer " not in block_text


class TestTraceLocalOnlyNoExternalFabric:
    """Trace emission must perform no network I/O."""

    def test_emit_live_trace_makes_no_network_calls(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TOK_TRACE", "1")
        monkeypatch.delenv("TOK_TRACE_FILE", raising=False)
        monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)
        session = _Session(tmp_path)

        calls: list[str] = []

        original_create_connection = getattr(__import__("socket"), "create_connection", None)

        def _mock_create_connection(*args, **kwargs):
            calls.append(f"create_connection({args})")
            if original_create_connection:
                return original_create_connection(*args, **kwargs)
            raise RuntimeError("Unexpected network call during trace emission")

        monkeypatch.setattr("socket.create_connection", _mock_create_connection)

        original_httpx_request = httpx.request

        def _mock_httpx_request(*args, **kwargs):
            calls.append(f"httpx.request({args})")
            return original_httpx_request(*args, **kwargs)

        monkeypatch.setattr("httpx.request", _mock_httpx_request)

        emit_live_trace(
            session,
            "request_prepared",
            trace_class="message",
            action="pass_through",
            result="ok",
            expectation="accept_pass_through",
            reason="test",
            direction="request",
            metadata={"compressed": True},
        )

        trace_files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(trace_files) >= 1, "Trace file should be written locally"

        network_calls = [c for c in calls if "create_connection" in c or "httpx.request" in c]
        assert len(network_calls) == 0, f"emit_live_trace should not make network calls, but got: {network_calls}"

    def test_live_trace_module_imports_no_network_libraries(self) -> None:
        import tok.spec.live_trace as live_trace_mod

        source = (
            live_trace_mod.__loader__.get_source(live_trace_mod.__name__)
            if hasattr(live_trace_mod.__loader__, "get_source")
            else ""
        )

        network_modules = {"requests", "aiohttp", "urllib3", "httpx", "websocket"}
        import_lines = [
            line.strip()
            for line in source.split("\n")
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]

        for line in import_lines:
            for mod in network_modules:
                assert f"import {mod}" not in line and f"from {mod}" not in line, (
                    f"live_trace.py should not import network library: {line}"
                )
