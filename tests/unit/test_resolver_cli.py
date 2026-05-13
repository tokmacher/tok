from __future__ import annotations

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


def test_resolver_help() -> None:
    result = runner.invoke(app, ["resolver", "--help"])
    assert result.exit_code == 0
    assert "Local resolver beta commands" in result.output
