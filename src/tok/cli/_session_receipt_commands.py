"""Experimental session receipt CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ._cli_support import _resolve_session_dir, console

session_receipt_app = typer.Typer(help="Generate local Tok session receipts", hidden=True)


@session_receipt_app.callback(invoke_without_command=True)
def session_receipt(
    ctx: typer.Context,
    latest: Annotated[bool, typer.Option("--latest", help="Use the latest local session with receipts.")] = False,
    session_dir: Annotated[
        Path | None, typer.Option("--session-dir", help="Session directory containing receipts.jsonl.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write receipt JSON to this path.")] = None,
) -> None:
    """Generate a local Tok session receipt."""
    if ctx.invoked_subcommand is not None:
        return
    resolved = _resolve_session_dir(session_dir=session_dir, latest=latest)
    if resolved is None:
        console.print("[red]No local session receipt source found.[/red]")
        raise typer.Exit(1)

    from tok.protocol.session_receipt import generate_session_receipt
    from tok.receipt import read_bridge_receipts

    receipts_path = resolved / "receipts.jsonl"
    savings_path = resolved / "savings_events.jsonl"
    receipts = read_bridge_receipts(receipts_path)
    session_id = receipts[-1].session_id if receipts else resolved.name
    receipt = generate_session_receipt(
        session_id=session_id,
        bridge_receipts_path=receipts_path,
        savings_events_path=savings_path if savings_path.exists() else None,
    )
    payload = receipt.model_dump(by_alias=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    console.print(f"session_id: {receipt.session.session_id}")
    console.print(f"turn_count: {receipt.session.turn_count}")
    console.print(f"savings_confidence: {receipt.savings_summary.confidence}")
    console.print(f"validation_level: {receipt.validation.level}")
    if output is not None:
        console.print(f"written: {output}")


def register(app: typer.Typer) -> None:
    app.add_typer(session_receipt_app, name="session-receipt", hidden=True)
