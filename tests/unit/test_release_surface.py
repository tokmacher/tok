from __future__ import annotations

from typer.testing import CliRunner

import tok
from tok.cli import app
from tok.release_surface import validate_release_surface

runner = CliRunner()


def test_release_surface_gate_passes_for_supported_bridge_story() -> None:
    help_output = runner.invoke(app, ["--help"]).output
    failures = validate_release_surface(
        exported_names=tok.__all__,
        cli_help_output=help_output,
        root_app=app,
    )

    assert failures == []


def test_experimental_root_helpers_are_not_in_the_supported_root_exports() -> None:
    assert "wrap" not in tok.__all__
    assert "process" not in tok.__all__
    assert "OrchestratorAdapter" not in tok.__all__
    assert "OpenAIChatAdapter" not in tok.__all__
    assert "TextLoopAdapter" not in tok.__all__
