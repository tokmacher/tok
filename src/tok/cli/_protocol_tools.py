"""Protocol conversion and parsing commands for the Tok CLI."""

from __future__ import annotations

import typer
from rich.console import Console

from tok.protocol.format_bridge import Bridge
from tok.protocol.parser import TokParser

console = Console()


def convert(
    payload: str,
    to: str = "tok",
    file: bool = False,
) -> None:
    """Convert JSON/Markdown into Tok (and vice versa)."""
    bridge = Bridge()
    if file:
        from pathlib import Path

        path = Path(payload)
        if not path.exists():
            console.print(f"[red]File not found: {payload}[/red]")
            raise typer.Exit(1)
        payload = path.read_text()

    to = to.lower()
    if to == "tok":
        console.print(bridge.detect_and_convert(payload))
    elif to == "json":
        console.print(bridge.to_json(payload))
    elif to == "md":
        console.print(bridge.to_md(payload))
    else:
        console.print(f"[red]Unknown target format: {to}[/red]")
        raise typer.Exit(1)


def parse(payload: str, file: bool = False) -> None:
    """Parse Tok markup and show AST nodes."""
    import json

    if file:
        from pathlib import Path

        path = Path(payload)
        if not path.exists():
            console.print(f"[red]File not found: {payload}[/red]")
            raise typer.Exit(1)
        payload = path.read_text()

    parser = TokParser()
    nodes = parser.parse(payload)
    from tok.protocol.parser import tok_to_dict

    console.print(json.dumps([tok_to_dict(node) for node in nodes], indent=2))
