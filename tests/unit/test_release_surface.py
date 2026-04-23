from __future__ import annotations

import pytest
from typer.testing import CliRunner

import tok
from tok.cli import app
from tok.release_surface import (
    CANDIDATE_PENDING_PROOF,
    EXPERIMENTAL_ROOT_EXPORTS,
    SUPPORTED_ROOT_EXPORTS,
    validate_release_surface,
)

runner = CliRunner()


def test_release_surface_gate_passes_for_supported_bridge_story() -> None:
    help_output = runner.invoke(app, ["--help"]).output
    failures = validate_release_surface(
        exported_names=tok.__all__,
        cli_help_output=help_output,
        root_app=app,
    )

    assert failures == []


def test_supported_root_exports_match_actual_importability() -> None:
    for name in SUPPORTED_ROOT_EXPORTS:
        obj = getattr(tok, name)
        assert obj is not None


def test_unsupported_root_imports_fail() -> None:
    for name in EXPERIMENTAL_ROOT_EXPORTS:
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(tok, name)
    for name in CANDIDATE_PENDING_PROOF:
        if name not in SUPPORTED_ROOT_EXPORTS:
            with pytest.raises(AttributeError, match="has no attribute"):
                getattr(tok, name)


def test_experimental_root_helpers_are_not_in_the_supported_root_exports() -> None:
    assert tok.__all__ == list(SUPPORTED_ROOT_EXPORTS)
    assert "wrap" not in tok.__all__
    assert "process" not in tok.__all__
    assert "OrchestratorAdapter" not in tok.__all__
    assert "OpenAIChatAdapter" not in tok.__all__
    assert "TextLoopAdapter" not in tok.__all__
