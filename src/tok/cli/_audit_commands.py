"""Trace audit command."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from tok.spec.trace import audit_trace_file

from ._cli_support import console, memory_root

FIXTURE_FILE_ARG = typer.Argument(None, help="Path to a Tok Trace v0.1 fixture or live JSONL trace file.")
JSON_OUTPUT_OPT = typer.Option(False, "--json", help="Emit machine-readable audit results.")
LATEST_OPT = typer.Option(False, "--latest", help="Audit the newest trace in the active .tok/traces directory.")


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
                console.print("[red]No trace files found in the active .tok/traces directory.[/red]")
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
            _print_live_trace_receipt(trace_file, payload)
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
        trace_dir = memory_root() / "traces"
        candidates = sorted(trace_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
    return fixture_file


def _print_live_trace_receipt(trace_file: Path, payload: list[dict[str, Any]]) -> None:
    blocks = _load_live_trace_blocks(trace_file)
    if not blocks:
        return

    status_counts = Counter(str(entry["status"]) for entry in payload)
    exact_count = sum(1 for block in blocks if block.get("content", {}).get("exact") is True)
    non_exact_count = sum(1 for block in blocks if block.get("content", {}).get("exact") is False)
    artifact_count = sum(1 for block in blocks if block.get("content", {}).get("resolver_uri"))
    reasons = sorted(
        {reason for block in blocks if isinstance(reason := block.get("audit", {}).get("reason"), str) and reason}
    )

    console.print("")
    console.print("[bold]Trace receipt[/bold]")
    console.print(
        f"Blocks: {len(blocks)} | "
        f"Pass: {status_counts['pass']} | "
        f"Warn: {status_counts['warn']} | "
        f"Fail: {status_counts['fail']}"
    )
    console.print(f"Exact: {exact_count} | Non-exact: {non_exact_count} | Artifacts: {artifact_count}/{len(blocks)}")
    if reasons:
        console.print("Reasons: " + "; ".join(reasons))


def _load_live_trace_blocks(trace_file: Path) -> list[dict[str, Any]]:
    if trace_file.suffix != ".jsonl":
        return []

    blocks: list[dict[str, Any]] = []
    try:
        lines = trace_file.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(record, dict):
            return []
        block = record.get("block") if "block" in record else record
        if not isinstance(block, dict):
            return []
        extensions = block.get("extensions")
        if isinstance(extensions, dict) and "tok.live" in extensions:
            blocks.append(block)
    return blocks
