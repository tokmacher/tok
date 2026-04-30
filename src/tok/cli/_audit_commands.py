"""Trace audit command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tok.spec.trace_v0_1 import audit_trace_file

from ._cli_support import console

FIXTURE_FILE_ARG = typer.Argument(None, help="Path to a Tok Trace v0.1 fixture or live JSONL trace file.")
JSON_OUTPUT_OPT = typer.Option(False, "--json", help="Emit machine-readable audit results.")
LATEST_OPT = typer.Option(False, "--latest", help="Audit the newest trace in ~/.tok/traces.")


def register(app: typer.Typer) -> None:
    """Register trace audit commands."""

    @app.command("audit")
    def audit(
        fixture_file: Path | None = FIXTURE_FILE_ARG,
        latest: bool = LATEST_OPT,
        json_output: bool = JSON_OUTPUT_OPT,
    ) -> None:
        """Audit Tok Trace v0.1 draft fixtures or live bridge traces."""
        trace_file = _resolve_audit_path(fixture_file, latest=latest)
        if trace_file is None:
            if latest:
                console.print("[red]No trace files found in ~/.tok/traces.[/red]")
            else:
                console.print("[red]Provide a trace file path or use --latest.[/red]")
            raise typer.Exit(5)
        if not trace_file.exists():
            console.print(f"[red]Trace file not found: {trace_file}[/red]")
            raise typer.Exit(5)

        results = audit_trace_file(trace_file)
        payload = [
            {
                "id": result.id,
                "status": result.status,
                "errors": list(result.errors),
                "summary": result.summary,
            }
            for result in results
        ]

        if json_output:
            print(json.dumps(payload, indent=2))
        else:
            for result in results:
                if result.status == "pass":
                    style = "green"
                elif result.status == "warn":
                    style = "yellow"
                else:
                    style = "red"
                suffix = f" {', '.join(result.errors)}" if result.errors else ""
                console.print(f"[{style}]{result.status.upper()}[/{style}] {result.id}{suffix}")
            if any(result.status == "warn" and "missing_identifiable" in result.errors for result in results):
                console.print(
                    "[yellow]Hint:[/yellow] metadata-only live traces warn when artifacts are not captured. "
                    "Use TOK_TRACE_CAPTURE_ARTIFACTS=1 for sanitized metadata artifact checks."
                )

        if any(result.status == "fail" for result in results):
            raise typer.Exit(1)
        if any(result.status == "warn" for result in results):
            raise typer.Exit(2)


def _resolve_audit_path(fixture_file: Path | None, *, latest: bool) -> Path | None:
    if latest and fixture_file is not None:
        console.print("[red]Use either a trace file path or --latest, not both.[/red]")
        raise typer.Exit(5)
    if latest:
        trace_dir = Path.home() / ".tok" / "traces"
        candidates = sorted(trace_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
    return fixture_file
