"""CLI commands for memory management and capture analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import typer

from tok.analysis.evidence_review import (
    build_coverage_report,
    load_stress_evidence,
    rank_candidates,
    review_capture_dir,
    summarize_capture_file,
)
from tok.runtime.memory.bridge_memory import (
    BridgeMemoryState,
    clean_system_context,
)

from ._cli_support import console, memory_root, render_stats_panel


def memory_snap(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Clear accumulated bridge memory while preserving essential state."""
    memory_file = memory_root() / "bridge_memory.tok"

    if not memory_file.exists():
        console.print("[yellow]No bridge memory file found — nothing to snap.[/yellow]")
        raise typer.Exit()

    if not yes:
        confirm = typer.confirm(
            "This will clear hot memory (cmds, errs, questions, next) "
            "while preserving goal/constraints/files in durable storage. Continue?"
        )
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit()

    state = BridgeMemoryState.from_tok(memory_file.read_text())

    preserved = {
        field: entries
        for field, entries in state.durable.items()
        if field in ("goal", "constraints", "files", "edited", "facts")
    }
    state.hot.clear()
    state.durable.clear()
    state.durable.update(preserved)
    state.turn = 0

    memory_root().mkdir(parents=True, exist_ok=True)
    memory_file.write_text(state.to_tok())

    kept_fields = sorted(preserved.keys())
    console.print(f"[green]Memory snapped.[/green] Preserved durable fields: {kept_fields or ['(none)']}")


def optimize_prompts(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Manual optimization of the current session's system prompt context."""
    memory_dir = memory_root()
    memory_file = memory_dir / "bridge_memory.tok"
    fallback_file = memory_dir / "memory.tok"

    if not memory_file.exists():
        console.print("[yellow]No bridge memory file found.[/yellow]")
        raise typer.Exit

    state = BridgeMemoryState.from_tok(memory_file.read_text())

    if fallback_file.exists():
        fallback_text = fallback_file.read_text()
        if len(fallback_text) > 2000:
            console.print(f"[yellow]Bloated fallback memory detected ({len(fallback_text)} chars).[/yellow]")
            if not yes and not typer.confirm("Apply automatic optimization?"):
                raise typer.Exit

            cleaned = clean_system_context(state, fallback_text)
            memory_file.write_text(state.to_tok())
            fallback_file.write_text(cast("str", cleaned) + "\n")
            console.print(f"[green]Optimized fallback memory to {len(cleaned)} chars.[/green]")
        else:
            console.print(f"[green]Fallback memory is lean ({len(fallback_text)} chars).[/green]")
    else:
        console.print("[dim]No fallback memory file found.[/dim]")

    bloated_fields = []
    for field, entries in state.durable.items():
        for entry in entries:
            if len(entry.value) > 500:
                bloated_fields.append((field, entry))

    if bloated_fields:
        console.print(f"[yellow]Found {len(bloated_fields)} bloated fields in durable memory.[/yellow]")
        if yes or typer.confirm("Compress bloated fields?"):
            from tok.compression import compress_user_prompt

            for field, entry in bloated_fields:
                old_len = len(entry.value)
                entry.value = compress_user_prompt(entry.value)
                console.print(f"  - {field}: {old_len} -> {len(entry.value)} chars")
            memory_file.write_text(state.to_tok())
            console.print("[green]Durable memory optimized.[/green]")
    else:
        console.print("[green]Durable memory is already optimal.[/green]")


def capture_summary(
    session_file: Annotated[str, typer.Argument(help="Path to .jsonl capture file")],
) -> None:
    """Summarize a captured bridge session without mutating any artifacts."""
    capture_path = Path(session_file)
    if not capture_path.exists():
        console.print(f"[red]File not found: {session_file}[/red]")
        raise typer.Exit(1)
    summary = summarize_capture_file(capture_path)

    rows = [
        ("Requests", str(summary["request_count"])),
        ("Dominant model", str(summary["dominant_model"])),
        (
            "Baseline-only",
            "yes" if summary["verdict"] == "investigate" else "no",
        ),
        ("Fallback events", str(summary["fallback_count"])),
        ("Repeat search", str(summary["repeat_search_count"])),
        ("Repeat file read", str(summary["repeat_file_read_count"])),
        ("Suggested verdict", str(summary["verdict"])),
    ]
    if str(summary["degradation_reason"]).strip():
        rows.append(("Last degradation reason", str(summary["degradation_reason"])))
    if float(summary["savings_pct"]) > 0:
        rows.append(("Session savings", f"{float(summary['savings_pct']):.1f}%"))

    console.print(
        render_stats_panel(
            "Capture Summary",
            headline=str(capture_path.name),
            headline_style="bold cyan",
            subhead="Read-only summary of captured bridge activity",
            rows=rows,
            border_style="cyan",
        )
    )


def capture_review(
    capture_dir: Annotated[str, typer.Argument(help="Directory containing captured session files")],
    verdict: Annotated[
        str | None,
        typer.Option("--verdict", help="Filter to clean, watch, or investigate sessions"),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Filter sessions by degradation-reason substring"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit the number of displayed sessions"),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write structured review output to a JSON file"),
    ] = None,
    candidates: Annotated[
        bool,
        typer.Option("--candidates", help="Show ranked promotion candidates"),
    ] = False,
    coverage: Annotated[
        bool,
        typer.Option(
            "--coverage",
            help="Show replay-coverage gaps for the reviewed evidence",
        ),
    ] = False,
    stress_dir: Annotated[
        Path | None,
        typer.Option("--stress-dir", help="Optional stress-language artifact directory"),
    ] = None,
    fixtures_dir: Annotated[
        Path,
        typer.Option("--fixtures-dir", help="Replay-fixture metadata directory"),
    ] = Path("tests/fixtures/replay"),
    gate_config: Annotated[
        Path,
        typer.Option(
            "--gate-config",
            help="Gate-config JSON for required fixture coverage",
        ),
    ] = Path("gate-config.json"),
) -> None:
    """Review captured real sessions in aggregate and rank replay-promotion candidates."""
    review = review_capture_dir(
        Path(capture_dir),
        verdict=verdict,
        reason_substring=reason,
        limit=limit,
    )
    candidate_rows = rank_candidates(review["sessions"])
    stress_evidence = load_stress_evidence(stress_dir) if stress_dir else None
    coverage_rows = (
        build_coverage_report(
            review["sessions"],
            stress_evidence=stress_evidence,
            replay_dir=fixtures_dir,
            gate_config_path=gate_config,
        )
        if coverage
        else []
    )

    console.print(
        render_stats_panel(
            "Capture Review",
            headline=f"{review['aggregate']['total_sessions']} sessions reviewed",
            headline_style="bold cyan",
            subhead=(
                f"clean={review['aggregate']['verdict_counts']['clean']}  "
                f"watch={review['aggregate']['verdict_counts']['watch']}  "
                f"investigate={review['aggregate']['verdict_counts']['investigate']}"
            ),
            rows=[
                (
                    "Fallback sessions",
                    str(review["aggregate"]["sessions_with_fallback_activity"]),
                ),
                (
                    "Reacquisition sessions",
                    str(review["aggregate"]["sessions_with_reacquisition_pressure"]),
                ),
            ],
            border_style="cyan",
        )
    )

    if review["aggregate"]["top_degradation_reasons"]:
        console.print("[bold]Top degradation reasons:[/bold]")
        for item in review["aggregate"]["top_degradation_reasons"][:5]:
            console.print(f"  - {item['reason']} ({item['count']})")

    if review["sessions"]:
        from rich.table import Table

        table = Table(title="Captured Sessions")
        table.add_column("Session")
        table.add_column("Model")
        table.add_column("Verdict")
        table.add_column("Fallbacks", justify="right")
        table.add_column("Repeat Search", justify="right")
        table.add_column("Repeat Read", justify="right")
        table.add_column("Reason")
        for item in review["sessions"]:
            table.add_row(
                str(item["name"]),
                str(item["dominant_model"]),
                str(item["verdict"]),
                str(item["fallback_count"]),
                str(item["repeat_search_count"]),
                str(item["repeat_file_read_count"]),
                str(item["degradation_reason"] or "none"),
            )
        console.print(table)
    else:
        console.print("[dim]No captured sessions matched the requested filters.[/dim]")

    if candidates and candidate_rows:
        from rich.table import Table

        table = Table(title="Promotion Candidates")
        table.add_column("Candidate")
        table.add_column("Score", justify="right")
        table.add_column("Evidence", justify="right")
        table.add_column("Models")
        table.add_column("Signals")
        table.add_column("Next Action")
        for item in candidate_rows:
            table.add_row(
                str(item["candidate"]),
                str(item["score"]),
                str(item["evidence_count"]),
                ", ".join(item["affected_models"]) or "unknown",
                ", ".join(item["dominant_signals"]) or "none",
                str(item["recommended_next_action"]),
            )
        console.print(table)

    if coverage and coverage_rows:
        from rich.table import Table

        console.print(
            "[bold]Coverage candidates:[/bold] " + ", ".join(str(item["candidate"]) for item in coverage_rows)
        )
        table = Table(title="Evidence Coverage")
        table.add_column("Candidate")
        table.add_column("Real")
        table.add_column("Stress")
        table.add_column("Coverage")
        table.add_column("Next Action")
        for item in coverage_rows:
            table.add_row(
                str(item["candidate"]),
                "yes" if item["seen_in_real_sessions"] else "no",
                "yes" if item["seen_in_stress_harness"] else "no",
                str(item["coverage_status"]),
                str(item["recommended_next_action"]),
            )
        console.print(table)

    if json_out is not None:
        payload: dict[str, Any] = {
            "sessions": review["sessions"],
            "aggregate": review["aggregate"],
            "candidates": candidate_rows,
        }
        if stress_evidence is not None:
            payload["stress_evidence"] = stress_evidence
        if coverage:
            payload["coverage"] = coverage_rows
        json_out.write_text(__import__("json").dumps(payload, indent=2))
        console.print(f"[green]Wrote capture review:[/green] {json_out}")


def evidence_gap(
    capture_dir: Annotated[str, typer.Argument(help="Directory containing captured session files")],
    stress_dir: Annotated[
        Path | None,
        typer.Option("--stress-dir", help="Optional stress-language artifact directory"),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write structured coverage output to JSON"),
    ] = None,
    fixtures_dir: Annotated[
        Path,
        typer.Option("--fixtures-dir", help="Replay-fixture metadata directory"),
    ] = Path("tests/fixtures/replay"),
    gate_config: Annotated[
        Path,
        typer.Option(
            "--gate-config",
            help="Gate-config JSON for required fixture coverage",
        ),
    ] = Path("gate-config.json"),
) -> None:
    """Show replay-coverage gaps using captured sessions and optional stress evidence."""
    review = review_capture_dir(Path(capture_dir))
    stress_evidence = load_stress_evidence(stress_dir) if stress_dir else None
    coverage_rows = build_coverage_report(
        review["sessions"],
        stress_evidence=stress_evidence,
        replay_dir=fixtures_dir,
        gate_config_path=gate_config,
    )
    if not coverage_rows:
        console.print("[dim]No evidence classes found yet.[/dim]")
    else:
        from rich.table import Table

        table = Table(title="Evidence Gap")
        table.add_column("Candidate")
        table.add_column("Coverage")
        table.add_column("Required")
        table.add_column("Exploratory")
        table.add_column("Next Action")
        for item in coverage_rows:
            table.add_row(
                str(item["candidate"]),
                str(item["coverage_status"]),
                ", ".join(item["required_fixtures"]) or "none",
                ", ".join(item["exploratory_fixtures"]) or "none",
                str(item["recommended_next_action"]),
            )
        console.print(table)
    if json_out is not None:
        payload: dict[str, Any] = {"coverage": coverage_rows}
        if stress_evidence is not None:
            payload["stress_evidence"] = stress_evidence
        json_out.write_text(__import__("json").dumps(payload, indent=2))
        console.print(f"[green]Wrote evidence gap:[/green] {json_out}")


def register(app: typer.Typer) -> None:
    """Register memory commands with the CLI app."""
    app.command("memory-snap", hidden=True)(memory_snap)
    app.command("optimize-prompts", hidden=True)(optimize_prompts)
    app.command("capture-summary", hidden=True)(capture_summary)
    app.command("capture-review", hidden=True)(capture_review)
    app.command("evidence-gap", hidden=True)(evidence_gap)
