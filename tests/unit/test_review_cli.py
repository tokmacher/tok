"""Phase 0.2.4: hidden agent review CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.receipt import BridgeReceipt, append_bridge_receipt

runner = CliRunner()


def _write_receipts(session_dir: Path) -> Path:
    path = session_dir / "receipts.jsonl"
    append_bridge_receipt(
        path,
        BridgeReceipt(
            receipt_id="r1",
            session_id="s1",
            turn=1,
            step=1,
            request_policy="natural_first",
            compression_applied=True,
            evidence_summary={"exact_entries": 2, "non_exact_latest": 1},
            savings={"tokens_saved": 50},
            fallback=False,
        ),
    )
    append_bridge_receipt(
        path,
        BridgeReceipt(
            receipt_id="r2",
            session_id="s1",
            turn=2,
            step=2,
            request_policy="natural_first",
            compression_applied=False,
            evidence_summary={"exact_entries": 3, "reacquisition_required": 1},
            savings={"tokens_saved": 0},
            fallback=True,
        ),
    )
    return path


def test_review_command_hidden_from_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "review" not in result.output


def test_review_session_json_from_session_dir(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "s1"
    _write_receipts(session_dir)

    result = runner.invoke(app, ["review", "session", "--session-dir", str(session_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["receipt_count"] == 2
    assert payload["fallback_count"] == 1
    assert payload["tokens_saved"] == 50
    assert payload["latest_turn"] == 2


def test_review_turn_latest_json(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "s1"
    _write_receipts(session_dir)

    result = runner.invoke(app, ["review", "turn", "--latest", "--session-dir", str(session_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["receipt_id"] == "r2"
    assert payload["fallback"] is True


def test_review_evidence_json(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "s1"
    _write_receipts(session_dir)

    result = runner.invoke(app, ["review", "evidence", "exact_entries", "--session-dir", str(session_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["key"] == "exact_entries"
    assert payload["latest_value"] == 3


def test_review_empty_session_exits_cleanly(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "empty"
    session_dir.mkdir(parents=True)

    result = runner.invoke(app, ["review", "session", "--session-dir", str(session_dir), "--json"])

    assert result.exit_code == 1
    assert "No receipts found" in result.output
