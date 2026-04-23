"""CLI commands for managing the Tok bridge server."""

from __future__ import annotations

from typing import Annotated

import typer

from ._cli_support import LOG_FILE, console


def bridge_start(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 9090,
    keep_turns: Annotated[int, typer.Option("--keep-turns", help="Human turns to keep verbatim")] = 2,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    foreground: Annotated[bool, typer.Option("--foreground", "-f", help="Run in foreground")] = False,
    fail_open: Annotated[
        bool,
        typer.Option("--fail-open/--no-fail-open", help="Pass through on errors"),
    ] = True,
    capture: Annotated[
        bool,
        typer.Option(
            "--capture/--no-capture",
            help="Capture bridge sessions to the Tok sessions directory",
        ),
    ] = False,
    api_base: Annotated[
        str,
        typer.Option(
            "--api-base",
            help="Target API base URL (e.g., https://api.anthropic.com)",
        ),
    ] = "https://api.anthropic.com",
) -> None:
    """Start the Tok bridge server."""
    from ._bridge import bridge_start as bridge_start_command

    bridge_start_command(
        port=port,
        keep_turns=keep_turns,
        debug=debug,
        foreground=foreground,
        fail_open=fail_open,
        capture=capture,
        api_base=api_base,
    )


def bridge_stop(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force stop even when called from a bridged Claude session.",
        ),
    ] = False,
) -> None:
    """Stop the Tok bridge server."""
    from ._bridge import bridge_stop as bridge_stop_command

    bridge_stop_command(force=force)


def bridge_status() -> None:
    """Check bridge status."""
    from ._bridge import bridge_status as bridge_status_command

    bridge_status_command()


def bridge_logs(
    lines: int = typer.Argument(40, help="Number of lines to show"),
) -> None:
    """Tail the bridge log file."""
    if not LOG_FILE.exists():
        console.print("[yellow]No log file found[/yellow]")
        raise typer.Exit(1)

    content = LOG_FILE.read_text().splitlines()
    for line in content[-lines:]:
        # Strip legacy prefix if present to avoid confusing Rich markup
        line = line.removeprefix("[tok-bridge] ")
        console.print(line, markup=True)


def register(bridge_app: typer.Typer) -> None:
    """Register bridge commands with the CLI app."""
    bridge_app.command("start")(bridge_start)
    bridge_app.command("stop")(bridge_stop)
    bridge_app.command("status")(bridge_status)
    bridge_app.command("logs")(bridge_logs)
