"""CLI commands for installing Tok shell integration."""

from __future__ import annotations

from typing import Annotated

import typer

from ._cli_support import console


def install(
    wrap_claude: Annotated[
        bool,
        typer.Option(
            "--wrap-claude",
            help="Install the optional claude() shell wrapper that auto-routes through Tok.",
        ),
    ] = False,
    uninstall: Annotated[
        bool,
        typer.Option(
            "--uninstall",
            help="Remove previously installed shell integration",
        ),
    ] = False,
) -> None:
    """Manage optional Tok shell integration."""
    from tok.utils import shell_integration

    if wrap_claude and uninstall:
        console.print("[red]Cannot combine `--wrap-claude` with `--uninstall`.[/red]")
        raise typer.Exit(2)

    try:
        if uninstall:
            removed = shell_integration.uninstall()
            if removed:
                console.print(
                    "[yellow]Tok shell integration removed from:[/yellow] " + ", ".join(str(path) for path in removed)
                )
            else:
                console.print("[yellow]Tok shell integration was not present in ~/.zshrc or ~/.bashrc.[/yellow]")
        elif wrap_claude:
            rc_path = shell_integration.install()
            console.print(f"[green]✅ Tok shell integration installed in {rc_path}.[/green]")
            console.print("[dim]Reload your shell: source " + str(rc_path) + "[/dim]")
            console.print("[dim]Wrapper mode enabled: `claude` will auto-start Tok bridge routing in this shell.[/dim]")
            console.print("[dim]Next step: run `claude`, then `tok doctor`.[/dim]")
        else:
            removed = shell_integration.uninstall()
            if removed:
                console.print(
                    "[yellow]Removed legacy `claude()` Tok wrapper from:[/yellow] "
                    + ", ".join(str(path) for path in removed)
                )
            else:
                console.print("[green]Tok install complete.[/green]")
            console.print(
                "[dim]Default mode is explicit: run `tok bridge start`, then "
                "`ANTHROPIC_BASE_URL=http://localhost:9090 claude`.[/dim]"
            )
            console.print("[dim]Optional wrapper mode: run `tok install --wrap-claude` if you want auto-routing.[/dim]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def convert(
    payload: Annotated[str, typer.Argument(help="Input text or JSON to convert")],
    to: Annotated[str, typer.Option("--to", help="Target format: tok | json | md")] = "tok",
    file: Annotated[
        bool,
        typer.Option(
            "--file",
            help="Treat payload as a file path instead of literal text",
        ),
    ] = False,
) -> None:
    """Convert JSON/Markdown into Tok (and vice versa)."""
    from ._protocol_tools import convert as convert_command

    convert_command(payload, to=to, file=file)


def parse(
    payload: Annotated[str, typer.Argument(help="Tok document or file path to parse")],
    file: Annotated[bool, typer.Option("--file", help="Treat payload as a file path")] = False,
) -> None:
    """Parse Tok markup and show the AST nodes."""
    from ._protocol_tools import parse as parse_command

    parse_command(payload, file=file)


def register(app: typer.Typer) -> None:
    """Register install commands with the CLI app."""
    app.command()(install)
    app.command(hidden=True)(convert)
    app.command(hidden=True)(parse)
