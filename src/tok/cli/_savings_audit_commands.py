"""Savings audit CLI command (hidden/experimental in 0.2.2).

Reads per-request savings events and checks invariants from
docs/savings-accounting.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ._cli_support import console, json_envelope

_EVENTS_FILENAME = "savings_events.jsonl"

_EVENTS_FILE_ARG = typer.Argument(..., help="Path to savings_events.jsonl file.")
_JSON_OUTPUT_OPT = typer.Option(False, "--json", help="Emit machine-readable output.")


def _audit_events(events: list[Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (summary_dict, violations) for a list of SavingsEvent objects."""
    from tok.utils.savings_reconstruction import reconstruct_session_summary

    summary = reconstruct_session_summary(events)
    violations: list[str] = []

    seen_ids: set[str] = set()
    for ev in events:
        # Duplicate event IDs
        if ev.event_id in seen_ids:
            violations.append(f"duplicate_event_id:{ev.event_id}")
        seen_ids.add(ev.event_id)

        # Negative token savings
        if ev.input_tokens_saved < 0:
            violations.append(f"negative_input_tokens_saved:{ev.event_id}")
        if ev.output_tokens_saved < 0:
            violations.append(f"negative_output_tokens_saved:{ev.event_id}")

        # Fallback requests must have zero headline savings
        if ev.fallback and ev.input_tokens_saved > 0:
            violations.append(f"fallback_with_nonzero_savings:{ev.event_id}")

    audit_data: dict[str, Any] = {
        "calls": summary.calls,
        "input_tokens_saved": summary.input_tokens_saved,
        "output_tokens_saved": summary.output_tokens_saved,
        "cost_saved_usd": round(summary.cost_saved_usd, 6),
        "fallback_count": summary.fallback_count,
        "degraded_count": summary.degraded_count,
        "violations": violations,
        "violation_count": len(violations),
    }
    return audit_data, violations


def register(app: typer.Typer) -> None:
    """Register savings audit command."""

    @app.command("savings-audit", hidden=True)
    def savings_audit(
        events_file: Path = _EVENTS_FILE_ARG,
        json_output: bool = _JSON_OUTPUT_OPT,
    ) -> None:
        """Audit per-request savings events for invariant violations. (experimental)"""
        from tok.utils.savings_event import read_savings_events

        if not events_file.exists():
            msg = f"Events file not found: {events_file}"
            if json_output:
                print(json.dumps(json_envelope("savings-audit", ok=False, status="error", data={"message": msg})))
            else:
                console.print(f"[red]{msg}[/red]")
            raise typer.Exit(5)

        events = read_savings_events(events_file)
        audit_data, violations = _audit_events(events)
        ok = len(violations) == 0

        if json_output:
            print(
                json.dumps(json_envelope("savings-audit", ok=ok, status="ok" if ok else "violations", data=audit_data))
            )
        else:
            _print_audit_report(audit_data, violations)

        if not ok:
            raise typer.Exit(1)


def _print_audit_report(audit_data: dict[str, Any], violations: list[str]) -> None:
    calls = audit_data.get("calls", 0)
    saved = audit_data.get("input_tokens_saved", 0)
    cost_saved = audit_data.get("cost_saved_usd", 0.0)
    violation_count = audit_data.get("violation_count", 0)

    console.print(f"  Calls:          {calls}")
    console.print(f"  Tokens saved:   {saved}")
    console.print(f"  Cost saved:     ${cost_saved:.6f}")
    if violations:
        console.print(f"\n[red]  {violation_count} violation(s) found:[/red]")
        for v in violations:
            console.print(f"    [red]• {v}[/red]")
    else:
        console.print("\n[green]  No violations found.[/green]")
