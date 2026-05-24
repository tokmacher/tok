"""TDD tests for tok.cli._mcp_commands — MCP CLI subcommand group.

Sub-stages:
  1. `tok mcp --help` shows start and install subcommands
  2. `tok mcp install --dry-run` exits 0 and prints valid JSON
  3. `tok mcp install --dry-run` output contains mcpServers.tok entry
  4. _write_mcp_config merges with existing mcpServers (does not clobber)
  5. _write_mcp_config with dry_run=True leaves filesystem unchanged
  6. _tok_mcp_entry always returns dict with "command" key
  7. _resolve_settings_path returns sensible default path
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stage 1: CLI group registration
# ---------------------------------------------------------------------------


def test_mcp_subcommands_reachable() -> None:
    """mcp is hidden but its subcommands must still be reachable."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0, result.output
    assert "start" in result.output
    assert "install" in result.output


# ---------------------------------------------------------------------------
# Stage 2 & 3: dry-run install produces valid config JSON
# ---------------------------------------------------------------------------


def test_mcp_install_dry_run_exits_zero(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    # --force bypasses the mcp-importability gate so the test is hermetic
    result = runner.invoke(app, ["mcp", "install", "--dry-run", "--force", "--settings", str(settings)])
    assert result.exit_code == 0, result.output


def test_mcp_install_dry_run_prints_mcp_servers(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    result = runner.invoke(app, ["mcp", "install", "--dry-run", "--force", "--settings", str(settings)])
    assert result.exit_code == 0, result.output
    assert "mcpServers" in result.output
    assert "tok" in result.output


def test_mcp_install_dry_run_output_is_parseable_json(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    result = runner.invoke(app, ["mcp", "install", "--dry-run", "--force", "--settings", str(settings)])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    json_start = next((i for i, line in enumerate(lines) if line.strip().startswith("{")), None)
    assert json_start is not None, f"No JSON found in output:\n{result.output}"
    json_text = "\n".join(lines[json_start:])
    parsed = json.loads(json_text)
    assert "mcpServers" in parsed
    assert "tok" in parsed["mcpServers"]
    entry = parsed["mcpServers"]["tok"]
    assert "command" in entry


def test_mcp_install_exits_one_when_mcp_not_installed(tmp_path: Path) -> None:
    """Without --force, install should refuse when mcp is not importable."""
    from tok.cli import _mcp_commands

    settings = tmp_path / "settings.json"
    orig = _mcp_commands._mcp_importable
    try:
        _mcp_commands._mcp_importable = lambda: False
        result = runner.invoke(app, ["mcp", "install", "--settings", str(settings)])
        assert result.exit_code == 1
        assert "mcp package is not installed" in result.output or "mcp" in result.output.lower()
    finally:
        _mcp_commands._mcp_importable = orig


# ---------------------------------------------------------------------------
# Stage 4 & 5: _write_mcp_config internals
# ---------------------------------------------------------------------------


def test_write_mcp_config_creates_file(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _write_mcp_config

    settings = tmp_path / "settings.json"
    _write_mcp_config(settings, dry_run=False)
    assert settings.exists()
    config = json.loads(settings.read_text())
    assert "mcpServers" in config
    assert "tok" in config["mcpServers"]


def test_write_mcp_config_merges_existing(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _write_mcp_config

    settings = tmp_path / "settings.json"
    existing = {"mcpServers": {"other-tool": {"command": "other-tool", "args": []}}}
    settings.write_text(json.dumps(existing))

    _write_mcp_config(settings, dry_run=False)
    config = json.loads(settings.read_text())
    assert "other-tool" in config["mcpServers"]
    assert "tok" in config["mcpServers"]


def test_write_mcp_config_dry_run_does_not_write(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _write_mcp_config

    settings = tmp_path / "settings.json"
    _write_mcp_config(settings, dry_run=True)
    assert not settings.exists()


def test_write_mcp_config_returns_dict(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _write_mcp_config

    settings = tmp_path / "settings.json"
    result = _write_mcp_config(settings, dry_run=True)
    assert isinstance(result, dict)
    assert "mcpServers" in result


def test_write_mcp_config_rejects_invalid_json(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _write_mcp_config

    settings = tmp_path / "settings.json"
    settings.write_text("not valid json at all {{")
    with pytest.raises(ValueError, match="not valid JSON"):
        _write_mcp_config(settings, dry_run=False)


def test_write_mcp_config_rejects_non_object_mcp_servers(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _write_mcp_config

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"mcpServers": ["list", "not", "object"]}))
    with pytest.raises(ValueError, match="not an object"):
        _write_mcp_config(settings, dry_run=False)


# ---------------------------------------------------------------------------
# Stage 6: _tok_mcp_entry
# ---------------------------------------------------------------------------


def test_tok_mcp_entry_has_command_key() -> None:
    from tok.cli._mcp_commands import _tok_mcp_entry

    entry = _tok_mcp_entry()
    assert "command" in entry
    assert isinstance(entry["command"], str)
    assert len(entry["command"]) > 0


def test_tok_mcp_entry_uses_tok_mcp_binary_when_available() -> None:
    from tok.cli._mcp_commands import _tok_mcp_entry

    with patch("shutil.which", return_value="/usr/local/bin/tok-mcp"):
        entry = _tok_mcp_entry()
    assert entry["command"] == "/usr/local/bin/tok-mcp"
    assert entry["args"] == []


def test_tok_mcp_entry_falls_back_to_python_module_when_binary_missing() -> None:
    from tok.cli._mcp_commands import _tok_mcp_entry

    with patch("shutil.which", return_value=None):
        entry = _tok_mcp_entry()
    # Must still produce a valid command (sys.executable path)
    assert "command" in entry
    assert "-m" in entry.get("args", [])
    assert "tok.mcp" in entry.get("args", [])


# ---------------------------------------------------------------------------
# Stage 7: _resolve_settings_path
# ---------------------------------------------------------------------------


def test_resolve_settings_path_returns_path(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _resolve_settings_path

    result = _resolve_settings_path(override=None)
    assert isinstance(result, Path)


def test_resolve_settings_path_uses_override() -> None:
    from tok.cli._mcp_commands import _resolve_settings_path

    override = Path("/tmp/custom_settings.json")
    result = _resolve_settings_path(override=override)
    assert result == override


def test_resolve_settings_path_prefers_claude_code_settings(tmp_path: Path) -> None:
    from tok.cli._mcp_commands import _resolve_settings_path

    # Simulate ~/.claude/settings.json existing
    fake_home = tmp_path
    claude_settings = fake_home / ".claude" / "settings.json"
    claude_settings.parent.mkdir()
    claude_settings.write_text("{}")

    with patch("pathlib.Path.home", return_value=fake_home):
        result = _resolve_settings_path(override=None)

    assert result == claude_settings
