from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs" / "spec" / "fixtures" / "tok_trace_v0_1_fixtures.json"

runner = CliRunner()


def test_hidden_audit_command_is_not_in_public_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "audit" not in result.output


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
