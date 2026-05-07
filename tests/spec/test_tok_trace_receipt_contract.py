from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.spec.live_trace import emit_live_trace

runner = CliRunner()


class _BridgeMemory:
    turn = 1


class _RuntimeSession:
    bridge_memory = _BridgeMemory()


class _Session:
    _active_session_key = "test-session"
    _live_trace_instance_id = "test-trace-instance"

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.runtime_session = _RuntimeSession()


def test_trace_receipt_contract_prints_expected_section(tmp_path, monkeypatch) -> None:
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
        metadata={"compressed": False},
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
        metadata={"output_saved_tokens": 0},
    )
    emit_live_trace(
        session,
        "fallback",
        trace_class="message",
        action="fallback",
        result="degraded",
        expectation="accept_fallback",
        reason="test",
        direction="response",
        metadata={"fallback": True},
    )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert trace_files, "Expected a live trace file to be written"
    trace_file = trace_files[0]

    result = runner.invoke(app, ["audit", str(trace_file)])
    assert result.exit_code in {0, 2}
    assert "Trace receipt" in result.output
    assert "Audit results:" in result.output
    assert "Live blocks:" in result.output


def test_verified_event_types_produce_live_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.delenv("TOK_TRACE_FILE", raising=False)
    monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)

    session = _Session(tmp_path)
    for event in ("request_prepared", "response_processed", "fallback"):
        emit_live_trace(
            session,
            event,
            trace_class="message",
            action="pass_through" if event != "fallback" else "fallback",
            result="ok" if event != "fallback" else "degraded",
            expectation="accept_pass_through" if event != "fallback" else "accept_fallback",
            reason="test",
            direction="request",
            metadata={},
        )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert trace_files
    content = trace_files[0].read_text()
    assert "request_prepared" in content
    assert "response_processed" in content
    assert "fallback" in content
