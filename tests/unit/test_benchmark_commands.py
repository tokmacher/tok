"""TDD tests for tok.cli._benchmark_commands — deterministic compression benchmark.

Sub-stages:
  1. _collect_fixtures returns sorted list of .jsonl files
  2. _fixture_to_report_entry returns correct schema for a real fixture
  3. _build_report returns dict with all required top-level keys
  4. tok benchmark run writes JSON output file
  5. tok benchmark report renders output (table or markdown)
  6. JSON report schema matches expected shape
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "replay"


# ---------------------------------------------------------------------------
# Stage 1: _collect_fixtures
# ---------------------------------------------------------------------------


def test_collect_fixtures_returns_list(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _collect_fixtures

    (tmp_path / "a.jsonl").write_text('{"messages": []}')
    (tmp_path / "b.jsonl").write_text('{"messages": []}')
    (tmp_path / "b.jsonl.meta.json").write_text("{}")  # must be excluded

    result = _collect_fixtures(tmp_path)
    names = [p.name for p in result]
    assert "a.jsonl" in names
    assert "b.jsonl" in names
    assert "b.jsonl.meta.json" not in names


def test_collect_fixtures_sorted(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _collect_fixtures

    for name in ["c.jsonl", "a.jsonl", "b.jsonl"]:
        (tmp_path / name).write_text('{"messages": []}')

    result = _collect_fixtures(tmp_path)
    assert [p.name for p in result] == ["a.jsonl", "b.jsonl", "c.jsonl"]


def test_collect_fixtures_empty_dir(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _collect_fixtures

    result = _collect_fixtures(tmp_path)
    assert result == []


# ---------------------------------------------------------------------------
# Stage 2: _fixture_to_report_entry
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="fixtures dir not present")
def test_fixture_to_report_entry_returns_expected_keys() -> None:
    from tok.cli._benchmark_commands import _fixture_to_report_entry

    fixture_files = sorted(FIXTURE_DIR.glob("*.jsonl"))
    if not fixture_files:
        pytest.skip("No fixture files found")

    entry = _fixture_to_report_entry(fixture_files[0])

    required_keys = {
        "input_saved_pct",
        "fallback_rate",
        "fallback_events",
        "turns",
        "strategy_breakdown",
        "total_before_tokens",
        "total_after_tokens",
        "input_saved_tokens",
        "fixture_kind",
    }
    assert required_keys.issubset(entry.keys()), f"Missing keys: {required_keys - entry.keys()}"


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="fixtures dir not present")
def test_fixture_to_report_entry_savings_pct_in_range() -> None:
    from tok.cli._benchmark_commands import _fixture_to_report_entry

    fixture_files = sorted(FIXTURE_DIR.glob("*.jsonl"))
    if not fixture_files:
        pytest.skip("No fixture files found")

    entry = _fixture_to_report_entry(fixture_files[0])
    assert 0.0 <= entry["input_saved_pct"] <= 100.0
    assert 0.0 <= entry["fallback_rate"] <= 1.0
    assert entry["turns"] >= 0
    assert entry["total_before_tokens"] >= entry["total_after_tokens"] >= 0


# ---------------------------------------------------------------------------
# Stage 3: _build_report
# ---------------------------------------------------------------------------


def test_build_report_returns_required_keys(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _build_report

    (tmp_path / "empty.jsonl").write_text("")

    report = _build_report(tmp_path)
    required = {"schema_version", "date", "python_version", "fixtures", "aggregate", "errors"}
    assert required.issubset(report.keys())


def test_build_report_schema_version_is_string(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _build_report

    report = _build_report(tmp_path)
    assert isinstance(report["schema_version"], str)


def test_build_report_with_no_fixtures_has_empty_aggregate(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _build_report

    report = _build_report(tmp_path)
    assert report["fixtures"] == {}
    assert report["aggregate"] == {}


def test_build_report_errors_captured_not_raised(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _build_report

    # A jsonl that will fail (bad content)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("this is not valid json at all\n")

    report = _build_report(tmp_path)
    # Should not raise; errors go in the errors dict
    assert isinstance(report["errors"], dict)


# ---------------------------------------------------------------------------
# Stage 4: tok benchmark run (CLI integration)
# ---------------------------------------------------------------------------


def test_benchmark_subcommands_reachable() -> None:
    """benchmark is hidden but its subcommands must still be reachable."""
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output
    assert "report" in result.output


def test_benchmark_run_creates_output_file(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "a.jsonl").write_text("")

    result = runner.invoke(app, ["benchmark", "run", "--fixtures", str(fixtures), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_benchmark_run_output_is_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()

    result = runner.invoke(app, ["benchmark", "run", "--fixtures", str(fixtures), "--output", str(output)])
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text())
    assert "schema_version" in data
    assert "aggregate" in data


def test_default_benchmark_report_path_is_user_writable_cache() -> None:
    """The default report path must not point into the repo/package install tree."""
    from tok.cli import _benchmark_commands

    report_path = _benchmark_commands._LATEST_REPORT
    assert report_path.name == "benchmark_latest.json"
    assert "tok" in report_path.parts
    assert report_path != REPO_ROOT / "results" / "benchmark_latest.json"
    assert "site-packages" not in report_path.parts


def test_benchmark_run_exits_one_when_fixture_dir_missing(tmp_path: Path) -> None:
    """Passing a nonexistent --fixtures dir must exit 1 with a clear message."""
    output = tmp_path / "bench.json"
    missing = tmp_path / "does_not_exist"

    result = runner.invoke(app, ["benchmark", "run", "--fixtures", str(missing), "--output", str(output)])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "does_not_exist" in result.output


def test_benchmark_run_exits_one_on_errors_by_default(tmp_path: Path) -> None:
    """By default, any fixture error must cause exit 1."""
    output = tmp_path / "bench.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "bad.jsonl").write_text("not json\n")

    result = runner.invoke(app, ["benchmark", "run", "--fixtures", str(fixtures), "--output", str(output)])
    assert result.exit_code == 1


def test_benchmark_run_allow_errors_exits_zero(tmp_path: Path) -> None:
    """--allow-errors lets errors be recorded without a non-zero exit."""
    output = tmp_path / "bench.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "bad.jsonl").write_text("not json\n")

    result = runner.invoke(
        app, ["benchmark", "run", "--fixtures", str(fixtures), "--output", str(output), "--allow-errors"]
    )
    assert result.exit_code == 0
    data = json.loads(output.read_text())
    assert data["errors"]  # errors captured, not swallowed


def test_benchmark_run_quiet_suppresses_benchmark_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """-q must silence logging emitted by the replay/compression pipeline."""
    output = tmp_path / "bench.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()

    def noisy_build_report(_fixture_dir: Path) -> dict:
        logging.getLogger("tok.compression").warning("benchmark noise should be hidden")
        return {
            "schema_version": "1",
            "date": "2026-01-01T00:00:00+00:00",
            "python_version": sys.version.split()[0],
            "fixtures": {},
            "aggregate": {},
            "errors": {},
        }

    monkeypatch.setattr("tok.cli._benchmark_commands._build_report", noisy_build_report)

    result = runner.invoke(app, ["benchmark", "run", "--fixtures", str(fixtures), "--output", str(output), "-q"])

    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_benchmark_run_quiet_disables_logging_during_report_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quiet mode must mute global logging only while benchmark fixtures run."""
    output = tmp_path / "bench.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    disable_calls: list[int] = []

    def fake_disable(level: int) -> None:
        disable_calls.append(level)

    def assert_logging_disabled(_fixture_dir: Path) -> dict:
        assert disable_calls == [logging.CRITICAL]
        return {
            "schema_version": "1",
            "date": "2026-01-01T00:00:00+00:00",
            "python_version": sys.version.split()[0],
            "fixtures": {},
            "aggregate": {},
            "errors": {},
        }

    monkeypatch.setattr(logging, "disable", fake_disable)
    monkeypatch.setattr("tok.cli._benchmark_commands._build_report", assert_logging_disabled)

    result = runner.invoke(app, ["benchmark", "run", "--fixtures", str(fixtures), "--output", str(output), "-q"])

    assert result.exit_code == 0, result.output
    assert disable_calls == [logging.CRITICAL, logging.NOTSET]


# ---------------------------------------------------------------------------
# Stage 5: tok benchmark report
# ---------------------------------------------------------------------------


def test_benchmark_report_from_json(tmp_path: Path) -> None:
    from tok.cli._benchmark_commands import _render_markdown

    report = {
        "schema_version": "1",
        "date": "2026-01-01T00:00:00+00:00",
        "python_version": sys.version.split()[0],
        "fixtures": {
            "test_fixture": {
                "input_saved_pct": 25.0,
                "fallback_events": 0,
                "fallback_rate": 0.0,
                "turns": 5,
                "strategy_breakdown": {"history_compression": 100},
                "total_before_tokens": 1000,
                "total_after_tokens": 750,
                "input_saved_tokens": 250,
                "fixture_kind": "compression",
            }
        },
        "aggregate": {
            "median_saved_pct": 25.0,
            "p90_saved_pct": 25.0,
            "total_tokens_saved": 250,
            "fallback_rate": 0.0,
            "fixture_count": 1,
            "error_count": 0,
        },
        "errors": {},
    }
    md = _render_markdown(report)
    assert "## " in md
    assert "25.0" in md
    assert "test_fixture" in md
    assert "tok/bench_fixtures/replay" in md
    assert "tests/fixtures/replay" not in md


def test_benchmark_report_cli_markdown_flag(tmp_path: Path) -> None:
    report_data = {
        "schema_version": "1",
        "date": "2026-01-01T00:00:00+00:00",
        "python_version": sys.version.split()[0],
        "fixtures": {},
        "aggregate": {},
        "errors": {},
    }
    input_file = tmp_path / "bench.json"
    input_file.write_text(json.dumps(report_data))

    result = runner.invoke(app, ["benchmark", "report", "--input", str(input_file), "--markdown"])
    assert result.exit_code == 0, result.output
    assert "Tok" in result.output


def test_benchmark_report_cli_json_flag(tmp_path: Path) -> None:
    report_data = {
        "schema_version": "1",
        "date": "2026-01-01T00:00:00+00:00",
        "python_version": sys.version.split()[0],
        "fixtures": {},
        "aggregate": {},
        "errors": {},
    }
    input_file = tmp_path / "bench.json"
    input_file.write_text(json.dumps(report_data))

    result = runner.invoke(app, ["benchmark", "report", "--input", str(input_file), "--json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert "schema_version" in parsed


# ---------------------------------------------------------------------------
# Stage 6: JSON schema regression
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="fixtures dir not present")
def test_build_report_aggregate_has_required_keys() -> None:
    from tok.cli._benchmark_commands import _build_report

    report = _build_report(FIXTURE_DIR)
    if not report["fixtures"]:
        pytest.skip("No fixtures processed")

    agg = report["aggregate"]
    required_agg_keys = {
        "median_saved_pct",
        "p90_saved_pct",
        "total_tokens_saved",
        "fallback_rate",
        "fixture_count",
        "error_count",
    }
    assert required_agg_keys.issubset(agg.keys()), f"Missing: {required_agg_keys - agg.keys()}"
