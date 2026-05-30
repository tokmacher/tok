"""Experimental handoff CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ._cli_support import _resolve_session_dir, console

handoff_app = typer.Typer(help="Export and inspect local Tok handoff packets", hidden=True)


@handoff_app.command("export")
def handoff_export(
    latest: Annotated[bool, typer.Option("--latest", help="Use the latest local session with receipts.")] = False,
    session_dir: Annotated[
        Path | None, typer.Option("--session-dir", help="Session directory containing receipts.jsonl.")
    ] = None,
    session_receipt: Annotated[
        Path | None, typer.Option("--session-receipt", help="Existing session receipt JSON path.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write handoff JSON to this path.")] = None,
    goal: Annotated[str, typer.Option("--goal", help="Current goal for the receiving agent.")] = "",
    source_agent: Annotated[str, typer.Option("--source-agent", help="Source agent label.")] = "unknown",
    target_agent: Annotated[str, typer.Option("--target-agent", help="Target agent label.")] = "unknown",
) -> None:
    """Export a local Tok handoff packet."""
    from tok.protocol.handoff import export_handoff
    from tok.receipt import read_bridge_receipts

    resolved = _resolve_session_dir(session_dir=session_dir, latest=latest)
    if session_receipt is None and resolved is None:
        console.print("[red]No local handoff source found.[/red]")
        raise typer.Exit(1)

    if session_receipt is not None:
        packet = export_handoff(
            session_receipt_path=session_receipt,
            source_agent=source_agent,
            target_agent=target_agent,
            goal=goal,
        )
    else:
        assert resolved is not None
        receipts_path = resolved / "receipts.jsonl"
        savings_path = resolved / "savings_events.jsonl"
        receipts = read_bridge_receipts(receipts_path)
        session_id = receipts[-1].session_id if receipts else resolved.name
        packet = export_handoff(
            session_id=session_id,
            bridge_receipts_path=receipts_path,
            savings_events_path=savings_path if savings_path.exists() else None,
            source_agent=source_agent,
            target_agent=target_agent,
            goal=goal,
        )

    payload = packet.model_dump(by_alias=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    console.print(f"handoff_id: {packet.handoff_id}")
    console.print(f"session_receipt: {packet.session_receipt.receipt_id}")
    console.print(f"required_reacquisitions: {len(packet.required_reacquisitions)}")
    if output is not None:
        console.print(f"written: {output}")


@handoff_app.command("inspect")
def handoff_inspect(
    path: Annotated[Path, typer.Argument(help="Handoff packet JSON path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect a local Tok handoff packet."""
    from tok.protocol.handoff import inspect_handoff

    result = inspect_handoff(path)
    payload = result.model_dump()
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
    else:
        console.print(result.summary)
        if result.required_reacquisitions:
            console.print(f"required_reacquisitions: {len(result.required_reacquisitions)}")
        for warning in result.warnings:
            console.print(f"[yellow]warning:[/yellow] {warning}")
        for error in result.errors:
            console.print(f"[red]error:[/red] {error}")
    if not result.passed:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    app.add_typer(handoff_app, name="handoff", hidden=True)


__all__ = ["handoff_app", "register"]
