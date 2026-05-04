"""One-command Claude Code launcher for Tok."""

from __future__ import annotations

import os
import subprocess
from typing import Annotated

import typer

from . import _bridge
from ._cli_support import console


def claude(
    ctx: typer.Context,
    port: Annotated[int, typer.Option("--port", "-p", help="Tok bridge port")] = 9090,
    api_base: Annotated[
        str | None,
        typer.Option(
            "--api-base",
            help="Target API base URL for the Tok bridge startup",
        ),
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Start the bridge with debug logging")] = False,
) -> None:
    """Start Tok if needed, then launch Claude Code through the bridge."""
    existing = _bridge.get_running_bridge_pid(port)
    if existing:
        console.print(f"[dim]Using existing Tok bridge on :{port} (PID {existing}).[/dim]")
    else:
        _bridge.bridge_start(port=port, api_base=api_base, debug=debug)

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://localhost:{port}"
    env["TOK_SELF_BRIDGED_SESSION"] = "1"
    env["TOK_BRIDGE_HOST"] = "localhost"
    env["TOK_BRIDGE_PORT"] = str(port)

    try:
        completed = subprocess.run(["claude", *ctx.args], env=env, check=False)
    except FileNotFoundError:
        console.print("[red]Claude Code executable not found: `claude`.[/red]")
        console.print("[dim]Install Claude Code first, then run `tok claude` again.[/dim]")
        raise typer.Exit(127) from None

    raise typer.Exit(completed.returncode)


def register(app: typer.Typer) -> None:
    """Register the one-command Claude launcher."""
    app.command(
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        }
    )(claude)
