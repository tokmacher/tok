from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.gateway import BridgeSession
from tok.spec.live_trace import emit_live_trace
from tok.spec.trace import audit_trace_file, canonical_payload_digest

runner = CliRunner()


class _BridgeMemory:
    turn = 3


class _RuntimeSession:
    bridge_memory = _BridgeMemory()


class _Session:
    _active_session_key = "test-session"
    _live_trace_instance_id = "test-trace-instance"

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
    assert record["extensions"]["tok.live"]["client_session_key"] == "test-session"
    assert record["extensions"]["tok.live"]["trace_instance_id"] == "test-trace-instance"
    assert record["envelope"]["payload_digest"] == canonical_payload_digest(record)

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


def test_live_trace_pass_through_ok_does_not_claim_fallback_expectation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")
    session = _Session(tmp_path)

    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="pass_through",
        result="ok",
        expectation="accept_fallback",
        reason="healthy pass-through request",
        metadata={"compressed": False},
    )

    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    record = json.loads(trace_file.read_text())

    assert record["observation"]["action"] == "pass_through"
    assert record["observation"]["result"] == "ok"
    assert record["audit"]["expectation"] == "accept_pass_through"


def test_live_trace_steps_are_monotonic_across_timestamp_wrap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    session = _Session(tmp_path)
    times = iter([1_999_999, 1_000_000])
    monkeypatch.setattr("tok.spec.live_trace.time.time_ns", lambda: next(times))

    for event in ("request_prepared", "response_processed"):
        emit_live_trace(
            session,
            event,
            trace_class="message",
            action="summary_reference",
            result="ok",
            expectation="accept_non_exact_reference",
            reason="metadata-only test",
            metadata={"event": event},
        )

    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    steps = [record["envelope"]["step"] for record in records]
    assert steps == sorted(steps)

    results = audit_trace_file(trace_file)
    assert not any(result.errors == ("out_of_order_trace_block",) for result in results)


def test_live_trace_distinguishes_bridge_restarts_in_shared_trace_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")
    trace_file = tmp_path / "shared.jsonl"
    monkeypatch.setenv("TOK_TRACE_FILE", str(trace_file))
    first = BridgeSession(memory_dir=tmp_path / "first")
    second = BridgeSession(memory_dir=tmp_path / "second")
    first._active_session_key = "hdr:same-client"
    second._active_session_key = "hdr:same-client"
    first.runtime_session.bridge_memory.turn = 2
    second.runtime_session.bridge_memory.turn = 1

    for session in (first, second):
        emit_live_trace(
            session,
            "request_prepared",
            trace_class="message",
            action="pass_through",
            result="ok",
            expectation="accept_pass_through",
            reason="shared trace file restart regression",
            metadata={"client": "same"},
        )

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert records[0]["extensions"]["tok.live"]["client_session_key"] == "hdr:same-client"
    assert records[1]["extensions"]["tok.live"]["client_session_key"] == "hdr:same-client"
    assert records[0]["envelope"]["session_id"] != records[1]["envelope"]["session_id"]
    assert all(record["envelope"]["payload_digest"] == canonical_payload_digest(record) for record in records)

    results = audit_trace_file(trace_file)
    assert all(result.status == "pass" for result in results)


def test_live_trace_sequence_audit_still_rejects_same_session_turn_regression(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")
    trace_file = tmp_path / "same-session-regression.jsonl"
    monkeypatch.setenv("TOK_TRACE_FILE", str(trace_file))
    session = BridgeSession(memory_dir=tmp_path / "bridge")
    session._active_session_key = "hdr:same-client"

    for turn in (2, 1):
        session.runtime_session.bridge_memory.turn = turn
        emit_live_trace(
            session,
            "request_prepared",
            trace_class="message",
            action="pass_through",
            result="ok",
            expectation="accept_pass_through",
            reason="same trace session must remain ordered",
            metadata={"turn": turn},
        )

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert records[0]["envelope"]["session_id"] == records[1]["envelope"]["session_id"]

    results = audit_trace_file(trace_file)
    assert any(result.errors == ("out_of_order_trace_block",) for result in results)


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


def test_audit_does_not_claim_full_session_coverage_for_partial_trace(monkeypatch, tmp_path: Path) -> None:
    """Guard: a trace starting at turn > 0 must not produce a full-session coverage claim.

    TOK_TRACE enabled mid-session writes records only from that turn onward.
    Audit must pass (records are individually valid) but must never emit
    'full_session_coverage' or similar misleading signal in errors or summary.
    """
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")

    class _LateBridgeMemory:
        turn = 5  # tracing started mid-session; turns 0-4 are absent

    class _LateRuntimeSession:
        bridge_memory = _LateBridgeMemory()

    class _LateSession:
        _active_session_key = "late-trace-session"

        def __init__(self, memory_dir: Path) -> None:
            self.memory_dir = memory_dir
            self.runtime_session = _LateRuntimeSession()

    session = _LateSession(tmp_path)
    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="mid-session trace start",
        metadata={"turn_at_trace_start": 5},
    )

    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    record = json.loads(trace_file.read_text())
    assert record["envelope"]["turn"] == 5, "fixture must start mid-session"

    results = audit_trace_file(trace_file)
    assert all(r.status in {"pass", "warn"} for r in results)

    # Guard against future code adding a misleading full-session coverage claim
    for r in results:
        assert "full_session_coverage" not in r.errors
        assert "full_session_coverage" not in r.summary

    # Guard: CLI --json output must not add full-session coverage signal either
    result = runner.invoke(app, ["audit", str(trace_file), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    for entry in payload:
        assert "full_session_coverage" not in entry.get("errors", [])
        assert "full_session_coverage" not in (entry.get("summary") or "")


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
