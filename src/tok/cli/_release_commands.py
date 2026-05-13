"""CLI commands for release management and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def stats(
    session: Annotated[
        bool,
        typer.Option(
            "--session",
            help="Show current session stats only (from tok_savings.tok)",
        ),
    ] = False,
    total: Annotated[
        bool,
        typer.Option(
            "--total",
            help="Show lifetime stats only (from global_savings.tok)",
        ),
    ] = False,
    breakdown: Annotated[
        bool,
        typer.Option("--breakdown", help="Show per-type compression breakdown"),
    ] = False,
    trends: Annotated[
        bool,
        typer.Option("--trends", help="Show recent trend summary from ledger"),
    ] = False,
    last_session: Annotated[
        bool,
        typer.Option(
            "--last-session",
            help="Show the most recent completed session from the lifetime ledger",
        ),
    ] = False,
    recent: Annotated[
        int | None,
        typer.Option(
            "--recent",
            help="Show an aggregate summary over the N most recent completed sessions",
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Show an aggregate summary over completed sessions since YYYY-MM-DD or ISO timestamp",
        ),
    ] = None,
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 5,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Reset lifetime stats (clear global ledger)"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON stats"),
    ] = False,
    detail: Annotated[
        bool,
        typer.Option("--detail", help="Show extra detail (bloat attribution, evidence forms, macro activity)"),
    ] = False,
) -> None:
    """Show token savings and fallback state."""
    from ._release import stats_command

    stats_command(
        session=session,
        total=total,
        breakdown=breakdown,
        trends=trends,
        last_session=last_session,
        recent=recent,
        since=since,
        window=window,
        reset=reset,
        json_output=json_output,
        detail=detail,
    )


def replay(
    session_file: Annotated[str, typer.Argument(help="Path to .jsonl capture file")],
    cost_per_mtok: Annotated[
        float,
        typer.Option(
            "--rate",
            help="Input cost per million tokens (default: sonnet rate)",
        ),
    ] = 3.0,
    gate: Annotated[
        bool,
        typer.Option(
            "--gate",
            help="Exit non-zero when replay shows weak savings or high pressure",
        ),
    ] = False,
) -> None:
    """Replay a captured session to measure compression savings offline."""
    from ._release import replay_command

    replay_command(
        session_file=session_file,
        cost_per_mtok=cost_per_mtok,
        gate=gate,
    )


def doctor(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show all behavior signals")] = False,
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            help="Print a pasteable environment report (safe to share)",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON diagnostics"),
    ] = False,
) -> None:
    """Check bridge health and runtime contract conformance."""
    from ._release import doctor_command

    doctor_command(verbose=verbose, report=report, json_output=json_output)


def gate_check(
    fixtures_dir: Annotated[Path, typer.Argument(help="Directory containing replay fixtures")],
    fixtures: Annotated[
        Path | None,
        typer.Option(
            "--fixtures",
            "-f",
            help="Optional JSON file listing fixtures to run",
        ),
    ] = None,
    export: Annotated[
        Path | None,
        typer.Option("--export", "-e", help="Path to export gate results JSON"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Gate-config JSON (default: gate-config.json in CWD)",
        ),
    ] = None,
    continue_on_error: Annotated[
        bool,
        typer.Option(
            "--continue-on-error",
            help="Continue running fixtures even if some fail",
        ),
    ] = False,
    fixture_set: Annotated[
        str | None,
        typer.Option("--set", help="Fixture set to use (feature, full, or redteam)"),
    ] = None,
    emit_metrics: Annotated[
        Path | None,
        typer.Option("--emit-metrics", help="Alias for --export (baseline_metrics.json)"),
    ] = None,
    stability_dir: Annotated[
        Path | None,
        typer.Option(
            "--stability-dir",
            help="Directory of *_stability.json files from live-benchmark runs. Checked against --required-benchmarks pass criteria.",
        ),
    ] = None,
    frontier_report: Annotated[
        Path | None,
        typer.Option(
            "--frontier-report",
            help="Compression frontier JSON report from `tok dev compression-frontier`. Validates that the current-head release lane is stable enough for release.",
        ),
    ] = None,
    benchmark_report: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-report",
            help="Production benchmark JSON report from `tok dev live-benchmark --program catalog|both`.",
        ),
    ] = None,
    required_benchmarks: Annotated[
        str,
        typer.Option(
            "--required-benchmarks",
            help="Comma-separated list of benchmark names that must be present and passing in --stability-dir (default: coding-loop-5,research-loop-5).",
        ),
    ] = "coding-loop-5,research-loop-5",
) -> None:
    """Run gate checks over a directory of replay fixtures."""
    from ._release import gate_check_command

    gate_check_command(
        fixtures_dir=fixtures_dir,
        fixtures=fixtures,
        export=export,
        config=config,
        continue_on_error=continue_on_error,
        fixture_set=fixture_set,
        emit_metrics=emit_metrics,
        stability_dir=stability_dir,
        frontier_report=frontier_report,
        benchmark_report=benchmark_report,
        required_benchmarks=required_benchmarks,
    )


def register(app: typer.Typer) -> None:
    """Register release commands with the CLI app."""
    app.command("stats")(stats)
    app.command(hidden=True)(replay)
    app.command()(doctor)
    app.command("gate-check", hidden=True)(gate_check)
