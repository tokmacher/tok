from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs" / "spec" / "fixtures" / "tok_trace_v0_1_fixtures.json"

runner = CliRunner()


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


def test_audit_latest_reports_missing_trace_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = runner.invoke(app, ["audit", "--latest"])

    assert result.exit_code == 5
    assert "No trace files found in ~/.tok/traces" in result.output


def test_audit_latest_uses_newest_trace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
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


def test_audit_rejects_path_and_latest_together(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
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
