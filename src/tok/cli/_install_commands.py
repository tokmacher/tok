from __future__ import annotations

from typing import Annotated

import typer

from ._cli_support import console


def install(
    uninstall: Annotated[
        bool,
        typer.Option(
            "--uninstall",
            help="Remove previously installed shell integration",
        ),
    ] = False,
) -> None:
    """Install or remove the Tok shell wrapper that adds `claude()`."""
    from .. import shell_integration

    try:
        if uninstall:
            removed = shell_integration.uninstall()
            if removed:
                console.print(
                    "[yellow]Tok shell integration removed from:[/yellow] "
                    + ", ".join(str(path) for path in removed)
                )
            else:
                console.print(
                    "[yellow]Tok shell integration was not present in ~/.zshrc or ~/.bashrc.[/yellow]"
                )
        else:
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


def convert(
    payload: Annotated[
        str, typer.Argument(help="Input text or JSON to convert")
    ],
    to: Annotated[
        str, typer.Option("--to", help="Target format: tok | json | md")
    ] = "tok",
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
    payload: Annotated[
        str, typer.Argument(help="Tok document or file path to parse")
    ],
    file: Annotated[
        bool, typer.Option("--file", help="Treat payload as a file path")
    ] = False,
) -> None:
    """Parse Tok markup and show the AST nodes."""
    from ._protocol_tools import parse as parse_command

    parse_command(payload, file=file)


def register(app: typer.Typer) -> None:
    app.command()(install)
    app.command(hidden=True)(convert)
    app.command(hidden=True)(parse)
