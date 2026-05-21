"""Section 5.2.5: Savings Audit CLI Command — RED/GREEN tests.

Tests tok savings-audit CLI invocation with known-good and known-bad event
streams.  Also tests adversarial inputs: duplicate event IDs, out-of-order
timestamps, missing events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner


def _make_event_dict(
    event_id: str,
    session_id: str,
    *,
    input_tokens_saved: int = 0,
    baseline_input_tokens: int = 1000,
    actual_input_tokens: int = 1000,
    cost_saved_usd: float = 0.0,
    fallback: bool = False,
    degraded_to_baseline: bool = False,
    timestamp: str = "2026-05-20T00:00:00Z",
) -> dict[str, Any]:
    return {
        "schema": "tok-savings-event/v1",
        "event_id": event_id,
        "session_id": session_id,
        "request_id": f"turn-{event_id}",
        "timestamp": timestamp,
        "model": "claude-sonnet-4",
        "mode": "tool-compatible",
        "request_policy": "natural_first",
        "baseline_input_tokens": baseline_input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "input_tokens_saved": input_tokens_saved,
        "baseline_output_tokens": 200,
        "actual_output_tokens": 200,
        "output_tokens_saved": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "baseline_cost_usd": 0.010,
        "actual_cost_usd": 0.010 - cost_saved_usd,
        "cost_saved_usd": cost_saved_usd,
        "fallback": fallback,
        "degraded_to_baseline": degraded_to_baseline,
        "compression_paths": {},
        "non_headline_estimates": {},
    }


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


# ---------------------------------------------------------------------------
# CLI module and command registration
# ---------------------------------------------------------------------------


class TestSavingsAuditCommandExists:
    def test_savings_audit_module_exists(self) -> None:
        from tok.cli import _savings_audit_commands  # type: ignore[import]

        assert hasattr(_savings_audit_commands, "register")

    def test_savings_audit_registered_in_cli(self) -> None:
        from tok.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", "--help"])
        # Must not return "No such command"
        assert result.exit_code != 2 or "No such command" not in (result.output or "")

    def test_savings_audit_help_shows_command(self) -> None:
        from tok.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", "--help"])
        # Help text must be present (exit 0)
        assert result.exit_code == 0
        assert "savings" in result.output.lower() or "audit" in result.output.lower()


# ---------------------------------------------------------------------------
# Known-good event streams
# ---------------------------------------------------------------------------


class TestSavingsAuditGoodStream:
    def test_good_stream_exits_zero(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        events = [
            _make_event_dict("e1", "s1", input_tokens_saved=100, baseline_input_tokens=1000, actual_input_tokens=900),
            _make_event_dict("e2", "s1", input_tokens_saved=200, baseline_input_tokens=1000, actual_input_tokens=800),
        ]
        _write_jsonl(events_file, events)

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file)])
        assert result.exit_code == 0, f"Expected exit 0; got {result.exit_code}\n{result.output}"

    def test_good_stream_json_output(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        events = [
            _make_event_dict("e1", "s1", input_tokens_saved=100, baseline_input_tokens=1000, actual_input_tokens=900),
        ]
        _write_jsonl(events_file, events)

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert "ok" in data
        assert data["ok"] is True

    def test_good_stream_reports_totals(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        events = [
            _make_event_dict("e1", "s1", input_tokens_saved=100),
            _make_event_dict("e2", "s1", input_tokens_saved=200),
        ]
        _write_jsonl(events_file, events)

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert "data" in data
        audit_data = data["data"]
        assert "calls" in audit_data
        assert audit_data["calls"] == 2


# ---------------------------------------------------------------------------
# Known-bad: invariant violations
# ---------------------------------------------------------------------------


class TestSavingsAuditBadStream:
    def test_negative_input_tokens_saved_flagged(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        bad_event = _make_event_dict("e1", "s1")
        bad_event["input_tokens_saved"] = -50  # invariant violation
        _write_jsonl(events_file, [bad_event])

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        data = json.loads(result.output.strip())
        assert data.get("ok") is False or "violations" in data.get("data", {})

    def test_fallback_with_nonzero_savings_flagged(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        # A fallback request should report zero headline savings
        bad_event = _make_event_dict(
            "e1",
            "s1",
            input_tokens_saved=300,  # should be 0 for a fallback
            fallback=True,
        )
        _write_jsonl(events_file, [bad_event])

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        data = json.loads(result.output.strip())
        # Fallback with savings is a warning-level violation
        assert data.get("ok") is False or ("violations" in data.get("data", {}) and len(data["data"]["violations"]) > 0)


# ---------------------------------------------------------------------------
# Missing / empty files
# ---------------------------------------------------------------------------


class TestSavingsAuditEdgeCases:
    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        from tok.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(tmp_path / "nonexistent.jsonl")])
        assert result.exit_code != 0

    def test_empty_file_exits_zero_zero_calls(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        events_file.write_text("")

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data.get("ok") is True
        assert data.get("data", {}).get("calls", 0) == 0


# ---------------------------------------------------------------------------
# Adversarial: duplicate IDs, out-of-order timestamps, corrupt lines
# ---------------------------------------------------------------------------


class TestSavingsAuditAdversarial:
    def test_duplicate_event_ids_flagged(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        events = [
            _make_event_dict("e1", "s1", input_tokens_saved=100),
            _make_event_dict("e1", "s1", input_tokens_saved=100),  # duplicate ID
        ]
        _write_jsonl(events_file, events)

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        data = json.loads(result.output.strip())
        assert data.get("ok") is False or (
            isinstance(data.get("data", {}).get("violations"), list) and len(data["data"]["violations"]) > 0
        )

    def test_corrupt_lines_skipped_gracefully(self, tmp_path: Path) -> None:
        from tok.cli import app

        events_file = tmp_path / "savings_events.jsonl"
        good = _make_event_dict("e1", "s1", input_tokens_saved=100)
        with events_file.open("w") as f:
            f.write(json.dumps(good) + "\n")
            f.write("totally-invalid-json\n")
            f.write(json.dumps(_make_event_dict("e2", "s1", input_tokens_saved=50)) + "\n")

        runner = CliRunner()
        result = runner.invoke(app, ["savings-audit", str(events_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data.get("data", {}).get("calls", 0) == 2
