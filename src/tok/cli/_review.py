"""Hidden agent review commands backed by local receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer

from ._cli_support import console

review_app = typer.Typer(help="Review local Tok session receipts", hidden=True)

_SESSION_DIR_OPT = typer.Option(None, "--session-dir", help="Session directory containing receipts.jsonl.")
_JSON_OPT = typer.Option(False, "--json", help="Emit machine-readable JSON.")
_LATEST_OPT = typer.Option(False, "--latest", help="Review the latest turn.")


@review_app.command("session")
def review_session(
    session_dir: Path | None = _SESSION_DIR_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Review a local session summary from receipts."""
    receipts = _load_receipts(session_dir)
    if not receipts:
        console.print("[red]No receipts found.[/red]")
        raise typer.Exit(1)
    payload = _session_summary(receipts)
    _emit(payload, json_output=json_output)


@review_app.command("turn")
def review_turn(
    latest: bool = _LATEST_OPT,
    session_dir: Path | None = _SESSION_DIR_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Review one turn from receipts."""
    receipts = _load_receipts(session_dir)
    if not receipts:
        console.print("[red]No receipts found.[/red]")
        raise typer.Exit(1)
    receipt = receipts[-1] if latest else receipts[0]
    _emit(receipt.to_dict(), json_output=json_output)


@review_app.command("evidence")
def review_evidence(
    key: str,
    session_dir: Path | None = _SESSION_DIR_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Review latest evidence counter for a key."""
    receipts = _load_receipts(session_dir)
    if not receipts:
        console.print("[red]No receipts found.[/red]")
        raise typer.Exit(1)
    latest_value: Any = None
    for receipt in receipts:
        if key in receipt.evidence_summary:
            latest_value = receipt.evidence_summary[key]
    payload = {"key": key, "latest_value": latest_value, "receipt_count": len(receipts)}
    _emit(payload, json_output=json_output)


def register(app: typer.Typer) -> None:
    app.add_typer(review_app, name="review", hidden=True)


def _load_receipts(session_dir: Path | None) -> list[Any]:
    from tok.receipt import read_bridge_receipts

    resolved = _resolve_session_dir(session_dir)
    if resolved is None:
        return []
    return read_bridge_receipts(resolved / "receipts.jsonl")


def _resolve_session_dir(session_dir: Path | None) -> Path | None:
    if session_dir is not None:
        return session_dir
    root = Path(os.getenv("TOK_DIR", str(Path.home() / ".tok"))) / "sessions"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "receipts.jsonl").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "receipts.jsonl").stat().st_mtime)


def _session_summary(receipts: list[Any]) -> dict[str, Any]:
    latest = receipts[-1]
    return {
        "session_id": latest.session_id,
        "receipt_count": len(receipts),
        "latest_turn": max(int(r.turn) for r in receipts),
        "fallback_count": sum(1 for r in receipts if r.fallback),
        "compression_count": sum(1 for r in receipts if r.compression_applied),
        "tokens_saved": sum(int((r.savings or {}).get("tokens_saved", 0) or 0) for r in receipts),
        "latest_evidence_summary": dict(latest.evidence_summary),
    }


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        console.print(json.dumps(payload, sort_keys=True))
        return
    for key, value in payload.items():
        console.print(f"{key}: {value}")


__all__ = ["register", "review_app"]
