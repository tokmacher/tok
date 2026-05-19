"""Trace audit command."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from tok.spec.trace import audit_trace_file

from ._cli_support import console, json_envelope, memory_root

FIXTURE_FILE_ARG = typer.Argument(None, help="Path to a Tok Trace v0.1 fixture or live JSONL trace file.")
JSON_OUTPUT_OPT = typer.Option(False, "--json", help="Emit machine-readable audit results.")
LATEST_OPT = typer.Option(False, "--latest", help="Audit the newest trace in the active .tok/traces directory.")


@dataclass(frozen=True)
class _LiveTraceReceipt:
    blocks: list[dict[str, Any]]
    skipped_records: int = 0


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
            msg = (
                "No trace files found in the active .tok/traces directory."
                if latest
                else "Provide a trace file path or use --latest."
            )
            if json_output:
                print(json.dumps(json_envelope("audit", ok=False, status="error", data={"message": msg})))
            else:
                console.print(f"[red]{msg}[/red]")
            raise typer.Exit(5)
        if not trace_file.exists():
            msg = f"Trace file not found: {trace_file}"
            if json_output:
                print(json.dumps(json_envelope("audit", ok=False, status="error", data={"message": msg})))
            else:
                console.print(f"[red]{msg}[/red]")
            raise typer.Exit(5)

        results = audit_trace_file(trace_file)
        live_receipt = _load_live_trace_blocks(trace_file)
        non_exact_live_ids = _non_exact_live_ids(live_receipt)
        payload = [
            {
                "id": result.id,
                "status": result.status,
                "errors": list(result.errors),
                "summary": _audit_result_summary(result.summary, result.id, non_exact_live_ids),
                "evidence_mode": _audit_evidence_mode(result.id, non_exact_live_ids),
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
                if result.id in non_exact_live_ids:
                    suffix += " (metadata-only non-exact)"
                console.print(f"[{style}]{result.status.upper()}[/{style}] {result.id}{suffix}")
            _print_live_trace_receipt(trace_file, payload, receipt=live_receipt)
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


def _non_exact_live_ids(receipt: _LiveTraceReceipt) -> set[str]:
    return {
        str(block.get("envelope", {}).get("block_id", ""))
        for block in receipt.blocks
        if block.get("content", {}).get("exact") is False
    }


def _audit_evidence_mode(result_id: str, non_exact_live_ids: set[str]) -> str:
    if result_id in non_exact_live_ids:
        return "metadata-only non-exact"
    return "trace-validated"


def _audit_result_summary(summary: str, result_id: str, non_exact_live_ids: set[str]) -> str:
    if summary:
        return summary
    if result_id in non_exact_live_ids:
        return "metadata-only non-exact"
    return ""


def _print_live_trace_receipt(
    trace_file: Path,
    payload: list[dict[str, Any]],
    *,
    receipt: _LiveTraceReceipt | None = None,
) -> None:
    receipt = receipt if receipt is not None else _load_live_trace_blocks(trace_file)
    blocks = receipt.blocks
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
        f"Audit results: {len(payload)} | "
        f"Pass: {status_counts['pass']} | "
        f"Warn: {status_counts['warn']} | "
        f"Fail: {status_counts['fail']}"
    )
    artifact_label = "Metadata artifacts" if exact_count == 0 and non_exact_count else "Artifacts"
    console.print(
        f"Live blocks: {len(blocks)} | "
        f"Exact: {exact_count} | "
        f"Non-exact: {non_exact_count} | "
        f"{artifact_label}: {artifact_count}/{len(blocks)}"
    )
    if receipt.skipped_records:
        console.print(f"Skipped receipt records: {receipt.skipped_records}")
    if reasons:
        console.print("Reasons: " + "; ".join(reasons))


def _load_live_trace_blocks(trace_file: Path) -> _LiveTraceReceipt:
    if trace_file.suffix != ".jsonl":
        return _LiveTraceReceipt([])

    blocks: list[dict[str, Any]] = []
    skipped_records = 0
    try:
        lines = trace_file.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return _LiveTraceReceipt([])

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped_records += 1
            continue
        if not isinstance(record, dict):
            skipped_records += 1
            continue
        block = record.get("block") if "block" in record else record
        if not isinstance(block, dict):
            skipped_records += 1
            continue
        extensions = block.get("extensions")
        if isinstance(extensions, dict) and "tok.live" in extensions:
            blocks.append(block)
        else:
            skipped_records += 1
    return _LiveTraceReceipt(blocks, skipped_records)
