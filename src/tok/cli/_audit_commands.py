"""Experimental trace audit command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tok.spec.trace_v0_1 import audit_trace_file

from ._cli_support import console

FIXTURE_FILE_ARG = typer.Argument(..., help="Path to a Tok Trace v0.1 fixture JSON file.")
JSON_OUTPUT_OPT = typer.Option(False, "--json", help="Emit machine-readable audit results.")


def register(app: typer.Typer) -> None:
    """Register hidden experimental audit commands."""

    @app.command("audit", hidden=True)
    def audit(
        fixture_file: Path = FIXTURE_FILE_ARG,
        json_output: bool = JSON_OUTPUT_OPT,
    ) -> None:
        """Validate draft Tok Trace fixture structure."""
        if not fixture_file.exists():
            console.print(f"[red]Fixture file not found: {fixture_file}[/red]")
            raise typer.Exit(5)

        results = audit_trace_file(fixture_file)
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
            console.print(json.dumps(payload, indent=2))
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

        if any(result.status == "fail" for result in results):
            raise typer.Exit(1)
        if any(result.status == "warn" for result in results):
            raise typer.Exit(2)
