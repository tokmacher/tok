from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.spec.live_trace import emit_live_trace
from tok.spec.trace_v0_1 import audit_trace_file

runner = CliRunner()


class _BridgeMemory:
    turn = 3


class _RuntimeSession:
    bridge_memory = _BridgeMemory()


class _Session:
    _active_session_key = "test-session"

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.runtime_session = _RuntimeSession()


def test_live_trace_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TOK_TRACE", raising=False)
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata-only test",
        metadata={"compressed": True},
    )

    assert not list((tmp_path / "traces").glob("*.jsonl"))


def test_live_trace_enabled_writes_auditable_jsonl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata-only test",
        metadata={"compressed": True, "input_saved_tokens": 42},
    )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    record = json.loads(trace_files[0].read_text())
    assert record["observation"]["key"] == "live:request_prepared"
    assert record["extensions"]["tok.live"]["metadata"]["input_saved_tokens"] == 42

    results = audit_trace_file(trace_files[0])
    assert len(results) == 1
    assert results[0].status == "warn"
    assert "missing_identifiable" in results[0].errors


def test_audit_command_accepts_live_jsonl_with_warnings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    session = _Session(tmp_path)
    emit_live_trace(
        session,
        "response_processed",
        trace_class="response",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata-only response trace",
        direction="response",
        metadata={"output_saved_tokens": 7},
    )
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))

    result = runner.invoke(app, ["audit", str(trace_file), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload[0]["status"] == "warn"


def test_live_trace_artifact_capture_produces_pass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata artifact test",
        metadata={"compressed": True, "input_saved_tokens": 11},
    )

    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    record = json.loads(trace_file.read_text())
    artifact_uri = record["content"]["resolver_uri"]
    assert artifact_uri.startswith("artifacts/")
    assert (trace_file.parent / artifact_uri).exists()

    results = audit_trace_file(trace_file)
    assert len(results) == 1
    assert results[0].status == "pass"


def test_audit_command_accepts_artifact_backed_live_jsonl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "response_processed",
        trace_class="response",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata artifact response trace",
        direction="response",
        metadata={"output_saved_tokens": 13},
    )
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))

    result = runner.invoke(app, ["audit", str(trace_file), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["status"] == "pass"


def test_live_trace_write_failure_is_non_fatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_FILE", str(tmp_path))
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "audit_warning",
        trace_class="system",
        action="pass_through",
        result="degraded",
        expectation="accept_fallback",
        reason="forced write failure test",
        metadata={"warning": True},
    )

    assert tmp_path.exists()
