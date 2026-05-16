from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.spec.live_trace import emit_live_trace

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs" / "spec" / "fixtures" / "trace_fixtures.json"
CLEAN_FIXTURE_PATH = ROOT / "docs" / "spec" / "fixtures" / "clean_trace_fixtures.json"
ADVERSARIAL_PACKS_PATH = ROOT / "docs" / "spec" / "fixtures" / "adversarial_packs.json"

runner = CliRunner()


class _BridgeMemory:
    turn = 4


class _RuntimeSession:
    bridge_memory = _BridgeMemory()


class _Session:
    _active_session_key = "audit-receipt-test"

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.runtime_session = _RuntimeSession()


def test_audit_command_is_in_public_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "audit" in result.output


def test_audit_command_reports_warnings_for_current_fixture_pack() -> None:
    result = runner.invoke(app, ["audit", str(FIXTURE_PATH), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    statuses = {entry["id"]: entry["status"] for entry in payload}
    assert statuses["missing_resolver_cache"] == "warn"
    assert statuses["unresolvable_fallback_required"] == "warn"
    assert statuses["malformed_block_rejection"] == "fail"


def test_audit_command_passes_clean_fixture_subset(tmp_path: Path) -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text())
    clean_fixtures = [
        fixture
        for fixture in fixtures
        if fixture["id"]
        in {
            "first_exact_file_observation",
            "unchanged_cached_tool_result",
            "delta_result",
            "fallback_raw_output",
            "repeated_search_reference",
            "skeleton_reference_non_exact",
            "summary_reference_non_exact",
        }
    ]
    clean_path = tmp_path / "clean_trace_fixtures.json"
    shutil.copytree(FIXTURE_PATH.parent / "artifacts", tmp_path / "artifacts")
    clean_path.write_text(json.dumps(clean_fixtures))

    result = runner.invoke(app, ["audit", str(clean_path)])

    assert result.exit_code == 0
    assert "PASS first_exact_file_observation" in result.output


def test_audit_command_passes_checked_in_clean_fixture_pack() -> None:
    result = runner.invoke(app, ["audit", str(CLEAN_FIXTURE_PATH)])

    assert result.exit_code == 0
    assert "PASS first_exact_file_observation" in result.output


def test_audit_human_output_includes_live_trace_receipt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOK_TRACE", "1")
    monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)
    session = _Session(tmp_path)
    emit_live_trace(
        session,
        "request_prepared",
        trace_class="message",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata-only request trace",
        metadata={"input_saved_tokens": 17},
    )
    monkeypatch.setenv("TOK_TRACE_CAPTURE_ARTIFACTS", "1")
    emit_live_trace(
        session,
        "response_processed",
        trace_class="response",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata artifact response trace",
        direction="response",
        metadata={"output_saved_tokens": 5},
    )
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))

    result = runner.invoke(app, ["audit", str(trace_file)])

    assert result.exit_code == 2
    assert "Trace receipt" in result.output
    assert "Audit results: 2" in result.output
    assert "Pass: 1" in result.output
    assert "Warn: 1" in result.output
    assert "Fail: 0" in result.output
    assert "Live blocks: 2" in result.output
    assert "Exact: 0" in result.output
    assert "Non-exact: 2" in result.output
    assert "Metadata artifacts: 1/2" in result.output
    assert "PASS " in result.output
    assert "(metadata-only non-exact)" in result.output
    assert "metadata-only request trace" in result.output


def test_audit_live_receipt_survives_malformed_jsonl_line(monkeypatch, tmp_path: Path) -> None:
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
        reason="metadata artifact request trace",
        metadata={"input_saved_tokens": 17},
    )
    emit_live_trace(
        session,
        "response_processed",
        trace_class="response",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata artifact response trace",
        direction="response",
        metadata={"output_saved_tokens": 5},
    )
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    trace_file.write_text(trace_file.read_text() + "{bad json\n")

    result = runner.invoke(app, ["audit", str(trace_file)])

    assert result.exit_code == 1
    assert "FAIL line:3 invalid_jsonl_line" in result.output
    assert "Trace receipt" in result.output
    assert "Audit results: 3" in result.output
    assert "Pass: 2" in result.output
    assert "Warn: 0" in result.output
    assert "Fail: 1" in result.output
    assert "Live blocks: 2" in result.output
    assert "Metadata artifacts: 2/2" in result.output
    assert "Skipped receipt records: 1" in result.output


def test_audit_live_receipt_skips_invalid_jsonl_records_without_hiding_valid_blocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
        reason="metadata artifact request trace",
        metadata={"input_saved_tokens": 17},
    )
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    trace_file.write_text(
        trace_file.read_text()
        + json.dumps(["not", "an", "object"])
        + "\n"
        + json.dumps({"block": "not-an-object"})
        + "\n"
    )

    result = runner.invoke(app, ["audit", str(trace_file)])

    assert result.exit_code == 1
    assert "FAIL line:2 jsonl_line_not_object" in result.output
    assert "FAIL line:3 missing_or_invalid_block" in result.output
    assert "Trace receipt" in result.output
    assert "Audit results: 3" in result.output
    assert "Live blocks: 1" in result.output
    assert "Metadata artifacts: 1/1" in result.output
    assert "Skipped receipt records: 2" in result.output


def test_audit_fixture_json_does_not_show_live_trace_receipt() -> None:
    result = runner.invoke(app, ["audit", str(CLEAN_FIXTURE_PATH)])

    assert result.exit_code == 0
    assert "Trace receipt" not in result.output


def test_audit_command_rejects_adversarial_pack_manifest_with_clear_error() -> None:
    result = runner.invoke(app, ["audit", str(ADVERSARIAL_PACKS_PATH)])

    assert result.exit_code == 1
    assert "adversarial_pack_manifest_not_a_trace_fixture" in result.output
    assert "invalid_jsonl_line" not in result.output


def test_audit_latest_reports_missing_trace_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TOK_PROJECT_DIR", raising=False)

    result = runner.invoke(app, ["audit", "--latest"])

    assert result.exit_code == 5
    assert "No trace files found in the active .tok/traces directory" in result.output


def test_audit_latest_uses_newest_trace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TOK_PROJECT_DIR", raising=False)
    trace_dir = tmp_path / ".tok" / "traces"
    trace_dir.mkdir(parents=True)
    old_trace = trace_dir / "old.jsonl"
    new_trace = trace_dir / "new.jsonl"
    old_trace.write_text("")
    new_trace.write_text("")
    os.utime(old_trace, (1, 1))
    os.utime(new_trace, (2, 2))

    result = runner.invoke(app, ["audit", "--latest", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload[0]["id"].endswith("new.jsonl")


def test_audit_latest_uses_project_trace_dir_when_configured(monkeypatch, tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("TOK_PROJECT_DIR", str(project_dir))
    home_trace_dir = home_dir / ".tok" / "traces"
    project_trace_dir = project_dir / ".tok" / "traces"
    home_trace_dir.mkdir(parents=True)
    project_trace_dir.mkdir(parents=True)
    home_trace = home_trace_dir / "home.jsonl"
    project_trace = project_trace_dir / "project.jsonl"
    home_trace.write_text("")
    project_trace.write_text("")
    os.utime(home_trace, (5, 5))
    os.utime(project_trace, (1, 1))

    result = runner.invoke(app, ["audit", "--latest", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload[0]["id"].endswith("project.jsonl")


def test_audit_rejects_path_and_latest_together(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TOK_PROJECT_DIR", raising=False)
    trace_dir = tmp_path / ".tok" / "traces"
    trace_dir.mkdir(parents=True)
    explicit_trace = tmp_path / "explicit.jsonl"
    latest_trace = trace_dir / "latest.jsonl"
    explicit_trace.write_text("")
    latest_trace.write_text("")

    result = runner.invoke(app, ["audit", str(explicit_trace), "--latest", "--json"])

    assert result.exit_code == 5
    assert "either a trace file path or --latest" in result.output


def test_audit_malformed_fixture_json_reports_failure_without_traceback(tmp_path: Path) -> None:
    trace_file = tmp_path / "bad_fixture.json"
    trace_file.write_text("[bad json")

    result = runner.invoke(app, ["audit", str(trace_file)])

    assert result.exit_code == 1
    assert "invalid_fixture_json" in result.output
    assert "Traceback" not in result.output


def test_audit_malformed_jsonl_line_reports_failure_without_traceback(tmp_path: Path) -> None:
    trace_file = tmp_path / "bad_trace.jsonl"
    trace_file.write_text("{bad json\n")

    result = runner.invoke(app, ["audit", str(trace_file)])

    assert result.exit_code == 1
    assert "invalid_jsonl_line" in result.output
    assert "Trace receipt" not in result.output
    assert "Traceback" not in result.output


def test_audit_non_utf8_trace_reports_failure_without_traceback(tmp_path: Path) -> None:
    trace_file = tmp_path / "bad_trace.jsonl"
    trace_file.write_bytes(b"\xff\xfe")

    result = runner.invoke(app, ["audit", str(trace_file)])

    assert result.exit_code == 1
    assert "trace_file_not_utf8" in result.output
    assert "Traceback" not in result.output


def test_audit_directory_path_reports_failure_without_traceback(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace_dir"
    trace_dir.mkdir()

    result = runner.invoke(app, ["audit", str(trace_dir)])

    assert result.exit_code == 1
    assert "trace_file_unreadable" in result.output
    assert "Traceback" not in result.output


def test_audit_exit_code_priority_clean_warn_and_fail(monkeypatch, tmp_path: Path) -> None:
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
        reason="metadata artifact request trace",
        metadata={"input_saved_tokens": 17},
    )
    clean_trace = next((tmp_path / "traces").glob("*.jsonl"))

    clean_result = runner.invoke(app, ["audit", str(clean_trace)])

    assert clean_result.exit_code == 0

    monkeypatch.delenv("TOK_TRACE_CAPTURE_ARTIFACTS", raising=False)
    warn_session = _Session(tmp_path / "warn")
    emit_live_trace(
        warn_session,
        "request_prepared",
        trace_class="message",
        action="summary_reference",
        result="ok",
        expectation="accept_non_exact_reference",
        reason="metadata-only request trace",
        metadata={"input_saved_tokens": 17},
    )
    warn_trace = next((tmp_path / "warn" / "traces").glob("*.jsonl"))

    warn_result = runner.invoke(app, ["audit", str(warn_trace)])

    assert warn_result.exit_code == 2

    clean_trace.write_text(clean_trace.read_text() + "{bad json\n")

    fail_result = runner.invoke(app, ["audit", str(clean_trace)])

    assert fail_result.exit_code == 1
