from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.cli._init_commands import _ensure_gitignore_entries, _maybe_create_env_file

runner = CliRunner()


def test_init_creates_tok_workspace_without_optional_files(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path), "--no-env", "--no-gitignore"])

    assert result.exit_code == 0
    assert (tmp_path / ".tok" / ".gitkeep").exists()
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".gitignore").exists()


def test_init_refuses_existing_workspace_without_force(tmp_path: Path) -> None:
    tok_dir = tmp_path / ".tok"
    tok_dir.mkdir()
    sentinel = tok_dir / ".gitkeep"
    sentinel.write_text("keep me")

    result = runner.invoke(app, ["init", str(tmp_path), "--no-env", "--no-gitignore"])

    assert result.exit_code == 1
    assert sentinel.read_text() == "keep me"


def test_maybe_create_env_file_leaves_existing_file_unchanged(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=1\n")

    created = _maybe_create_env_file(env_path)

    assert created is False
    assert env_path.read_text() == "EXISTING=1\n"


def test_gitignore_helper_repairs_partial_tok_section_without_duplicate_header(tmp_path: Path) -> None:
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("dist/\n\n# Tok\n.tok/\n")

    updated = _ensure_gitignore_entries(gitignore_path, entries=[".tok/", "telemetry.db"])

    content = gitignore_path.read_text()
    assert updated is True
    assert content.count("# Tok") == 1
    assert content.count(".tok/") == 1
    assert content.count("telemetry.db") == 1
