"""tok mcp — MCP server commands (Model Context Protocol)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from ._cli_support import console


def _settings_candidates() -> list[Path]:
    home = Path.home()
    return [
        home / ".claude" / "settings.json",
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        home / ".config" / "claude" / "claude_desktop_config.json",
    ]


def _resolve_settings_path(override: Path | None) -> Path:
    """Return the target settings file: explicit override, first existing candidate, or default."""
    if override is not None:
        return override
    candidates = _settings_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _mcp_importable() -> bool:
    """Return True if the mcp package can be imported in the current Python."""
    import importlib.util

    return importlib.util.find_spec("mcp") is not None


def _tok_mcp_entry() -> dict:
    """Build the mcpServers entry for Tok, falling back to python -m tok.mcp."""
    tok_mcp_bin = shutil.which("tok-mcp")
    if tok_mcp_bin:
        return {"command": tok_mcp_bin, "args": []}
    return {"command": sys.executable, "args": ["-m", "tok.mcp"]}


def _write_mcp_config(settings_path: Path, *, dry_run: bool) -> dict:
    """Inject tok into mcpServers in the settings file atomically. Returns the final config."""
    if settings_path.exists():
        try:
            config: dict = json.loads(settings_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Settings file is not valid JSON ({settings_path}): {exc}") from exc
        if not isinstance(config.get("mcpServers", {}), dict):
            raise ValueError(f"mcpServers in {settings_path} is not an object; refusing to overwrite")
    else:
        config = {}

    config.setdefault("mcpServers", {})
    config["mcpServers"]["tok"] = _tok_mcp_entry()

    if not dry_run:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve existing file mode so we don't change permissions on the user's config.
        existing_mode = settings_path.stat().st_mode if settings_path.exists() else 0o600
        # Atomic write: write to a temp file in the same directory, then rename.
        fd, tmp = tempfile.mkstemp(dir=settings_path.parent, prefix=".tok-mcp-install-", suffix=".tmp")
        try:
            os.chmod(tmp, existing_mode)
            with open(fd, "w") as fh:
                fh.write(json.dumps(config, indent=2) + "\n")
            Path(tmp).replace(settings_path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    return config


def register(app: typer.Typer) -> None:
    mcp_app = typer.Typer(help="MCP server commands (Model Context Protocol)", no_args_is_help=True)
    app.add_typer(mcp_app, name="mcp", hidden=True)

    @mcp_app.command("start")
    def mcp_start() -> None:
        """Start the Tok MCP server using stdio transport (for Claude Code)."""
        try:
            from tok.mcp.server import main

            main()
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    @mcp_app.command("install")
    def mcp_install(
        settings_file: Annotated[
            Path | None,
            typer.Option("--settings", help="Path to Claude Code settings.json (auto-detected if omitted)"),
        ] = None,
        dry_run: Annotated[bool, typer.Option("--dry-run", help="Print config without writing")] = False,
        force: Annotated[
            bool,
            typer.Option("--force", help="Register even if the mcp extra is not installed"),
        ] = False,
    ) -> None:
        """Register Tok as an MCP server in Claude Code settings.json."""
        if not _mcp_importable() and not force:
            console.print("[yellow]Warning: the mcp package is not installed.[/yellow]")
            console.print("  Install it first: [cyan]pip install tok-protocol[mcp][/cyan]")
            console.print("  Then re-run this command, or pass [cyan]--force[/cyan] to register anyway.")
            raise typer.Exit(1)

        path = _resolve_settings_path(settings_file)
        config = _write_mcp_config(path, dry_run=dry_run)
        if dry_run:
            console.print(json.dumps(config, indent=2))
        else:
            console.print(f"[green]Tok MCP server registered in {path}[/green]")
            console.print("Restart Claude Code for the change to take effect.")
