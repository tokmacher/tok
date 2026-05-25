"""Phase 0.2.4: agent contract verification automation."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


def test_verify_contract_passes_for_repo_contract() -> None:
    from tok.cli._contract import verify_contract

    result = verify_contract()
    assert result["ok"] is True
    assert result["violations"] == []


def test_verify_contract_catches_forbidden_help_text() -> None:
    from tok.cli._contract import verify_contract

    result = verify_contract(help_text="Tok provides protocol compliance certified by tok audit")
    assert result["ok"] is False
    assert any("forbidden_claim" in violation for violation in result["violations"])


def test_verify_contract_catches_deferred_layer_help_text() -> None:
    from tok.cli._contract import verify_contract

    result = verify_contract(help_text="Tok supports agent-to-agent exchange implemented today")
    assert result["ok"] is False
    assert any("deferred_layer_visible" in violation for violation in result["violations"])


def test_verify_contract_json_cli() -> None:
    result = runner.invoke(app, ["verify-contract", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["contract"]["schema"] == "tok-agent-contract/v0.1"


def test_verify_contract_hidden_from_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "verify-contract" not in result.output


def test_verify_contract_missing_contract_fails(tmp_path: Path) -> None:
    from tok.cli._contract import verify_contract

    missing = tmp_path / "missing.json"
    result = verify_contract(contract_path=missing)
    assert result["ok"] is False
    assert "contract_missing" in result["violations"]
