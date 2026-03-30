"""Install command handler for Tok CLI."""

from __future__ import annotations

import typer
from rich.console import Console

from .. import shell_integration

console = Console()


def install() -> None:
    """Install Tok shell helpers (tok doctor/swap commands)."""
    try:
        rc_path = shell_integration.install()
        console.print(
            f"[green]✅ Tok shell integration installed in {rc_path}.[/green]"
        )
        console.print(
            "[dim]Open a new shell or run: source " + str(rc_path) + "[/dim]"
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
