"""Install command handler for Tok CLI."""

from __future__ import annotations

import typer
from rich.console import Console

from tok.utils import shell_integration

console = Console()


def install() -> None:
    """Install optional Tok shell helper for claude() wrapper mode."""
    try:
        rc_path = shell_integration.install()
        console.print(f"[green]✅ Tok shell integration installed in {rc_path}.[/green]")
        console.print("[dim]Reload your shell: source " + str(rc_path) + "[/dim]")
        console.print("[dim]Wrapper mode enabled: `claude` will auto-route through Tok bridge.[/dim]")
        console.print("[dim]Next step: run `claude`, then `tok doctor`.[/dim]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
