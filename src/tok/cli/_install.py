"""Install command handler for Tok CLI."""

from __future__ import annotations

import typer
from rich.console import Console

from .. import shell_integration

console = Console()


def install() -> None:
    """Install the Tok shell helper that adds the claude() wrapper."""
    try:
        rc_path = shell_integration.install()
        console.print(
            f"[green]✅ Tok shell integration installed in {rc_path}.[/green]"
        )
        console.print(
            "[dim]Reload your shell: source " + str(rc_path) + "[/dim]"
        )
        console.print(
            "[dim]Next step: run `tok bridge start`, then `claude`, then `tok doctor`.[/dim]"
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
