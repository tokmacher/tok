from __future__ import annotations

"""Tok CLI — command-line interface for the Tok bridge and protocol tools."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, cast
from collections.abc import Mapping

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("TOK_LOG_LEVEL", "INFO").upper())

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..analysis.evidence_review import (
    build_coverage_report,
    load_stress_evidence,
    rank_candidates,
    review_capture_dir,
    summarize_capture_file,
)
from ..runtime.memory.bridge_memory import (
    BridgeMemoryState,
    clean_system_context,
)
from ..stats import SavingsTracker
from ._cli_support import (
    LOG_FILE,
    bridge_url,
    console,
    get_running_bridge_pid,
    memory_root,
    render_stats_panel,
    runtime_verdict,
    savings_headline,
    savings_style,
    savings_verdict,
    session_recommendation,
    session_signals_text,
    session_status_rows,
    status_border,
)
from ._dev import (
    dev_app,
    generate_fixture as dev_generate_fixture,
    live_benchmark as dev_live_benchmark,
    stress_language as dev_stress_language,
)
from ._metrics import (
    fallback as metrics_fallback,
    health as metrics_health,
    memory as metrics_memory,
    metrics_app,
    pressure as metrics_pressure,
    savings_trend as metrics_savings_trend,
)

app = typer.Typer(
    help="Tok — bridge-first CLI for Claude Code", add_completion=False
)
bridge_app = typer.Typer(help="Bridge-first workflow commands")
app.add_typer(bridge_app, name="bridge")
app.add_typer(metrics_app, name="metrics", hidden=True)
app.add_typer(dev_app, name="dev", hidden=True)


@app.command()
def install(
    uninstall: Annotated[
        bool,
        typer.Option(
            "--uninstall",
            help="Remove previously installed shell integration",
        ),
    ] = False,
) -> None:
    """Install or remove the Tok shell wrapper that adds `claude()`."""

    from .. import shell_integration

    try:
        if uninstall:
            removed = shell_integration.uninstall()
            if removed:
                console.print(
                    "[yellow]Tok shell integration removed from:[/yellow] "
                    + ", ".join(str(path) for path in removed)
                )
            else:
                console.print(
                    "[yellow]Tok shell integration was not present in ~/.zshrc or ~/.bashrc.[/yellow]"
                )
        else:
            rc_path = shell_integration.install()
            console.print(
                f"[green]✅ Tok shell integration installed in {rc_path}.[/green]"
            )
            console.print(
                "[dim]Reload your shell: source " + str(rc_path) + "[/dim]"
            )
            console.print(
                "[dim]Next step: run `tok bridge start`, then `claude`, then `tok doctor`.[/dim]"
            )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command(hidden=True)
def convert(
    payload: Annotated[
        str, typer.Argument(help="Input text or JSON to convert")
    ],
    to: Annotated[
        str, typer.Option("--to", help="Target format: tok | json | md")
    ] = "tok",
    file: Annotated[
        bool,
        typer.Option(
            "--file",
            help="Treat payload as a file path instead of literal text",
        ),
    ] = False,
) -> None:
    """Convert JSON/Markdown into Tok (and vice versa)."""
    from ._protocol_tools import convert as convert_command

    convert_command(payload, to=to, file=file)


@app.command(hidden=True)
def parse(
    payload: Annotated[
        str, typer.Argument(help="Tok document or file path to parse")
    ],
    file: Annotated[
        bool, typer.Option("--file", help="Treat payload as a file path")
    ] = False,
) -> None:
    """Parse Tok markup and show the AST nodes."""
    from ._protocol_tools import parse as parse_command

    parse_command(payload, file=file)


# ---------------------------------------------------------------------------
# Bridge commands
# ---------------------------------------------------------------------------


@bridge_app.command("start")
def bridge_start(
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to listen on")
    ] = 9090,
    keep_turns: Annotated[
        int, typer.Option("--keep-turns", help="Human turns to keep verbatim")
    ] = 2,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", "-f", help="Run in foreground")
    ] = False,
    fail_open: Annotated[
        bool,
        typer.Option(
            "--fail-open/--no-fail-open", help="Pass through on errors"
        ),
    ] = True,
    capture: Annotated[
        bool,
        typer.Option(
            "--capture/--no-capture",
            help="Capture bridge sessions to the Tok sessions directory",
        ),
    ] = False,
    api_base: Annotated[
        str,
        typer.Option(
            "--api-base",
            help="Target API base URL (e.g., https://api.anthropic.com)",
        ),
    ] = "https://api.anthropic.com",
) -> None:
    """Start the Tok bridge server."""
    from ._bridge import bridge_start as bridge_start_command

    bridge_start_command(
        port=port,
        keep_turns=keep_turns,
        debug=debug,
        foreground=foreground,
        fail_open=fail_open,
        capture=capture,
        api_base=api_base,
    )


@bridge_app.command("stop")
def bridge_stop() -> None:
    """Stop the Tok bridge server."""
    from ._bridge import bridge_stop as bridge_stop_command

    bridge_stop_command()


@bridge_app.command("status")
def bridge_status() -> None:
    """Check bridge status."""
    from ._bridge import bridge_status as bridge_status_command

    bridge_status_command()


@bridge_app.command("logs")
def bridge_logs(
    lines: int = typer.Argument(40, help="Number of lines to show"),
) -> None:
    """Tail the bridge log file."""
    if not LOG_FILE.exists():
        console.print("[yellow]No log file found[/yellow]")
        raise typer.Exit(1)

    content = LOG_FILE.read_text().splitlines()
    for line in content[-lines:]:
        console.print(line)


# ---------------------------------------------------------------------------
# Memory commands
# ---------------------------------------------------------------------------


@app.command("memory-snap", hidden=True)
def memory_snap(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Clear accumulated bridge memory while preserving essential state."""
    memory_file = memory_root() / "bridge_memory.tok"

    if not memory_file.exists():
        console.print(
            "[yellow]No bridge memory file found — nothing to snap.[/yellow]"
        )
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
    console.print(
        f"[green]Memory snapped.[/green] Preserved durable fields: {kept_fields or ['(none)']}"
    )


@app.command("optimize-prompts", hidden=True)
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
        raise typer.Exit()

    state = BridgeMemoryState.from_tok(memory_file.read_text())

    if fallback_file.exists():
        fallback_text = fallback_file.read_text()
        if len(fallback_text) > 2000:
            console.print(
                f"[yellow]Bloated fallback memory detected ({len(fallback_text)} chars).[/yellow]"
            )
            if not yes:
                if not typer.confirm("Apply automatic optimization?"):
                    raise typer.Exit()

            cleaned = clean_system_context(state, fallback_text)
            memory_file.write_text(state.to_tok())
            fallback_file.write_text(cast(str, cleaned) + "\n")
            console.print(
                f"[green]Optimized fallback memory to {len(cleaned)} chars.[/green]"
            )
        else:
            console.print(
                f"[green]Fallback memory is lean ({len(fallback_text)} chars).[/green]"
            )
    else:
        console.print("[dim]No fallback memory file found.[/dim]")

    bloated_fields = []
    for field, entries in state.durable.items():
        for entry in entries:
            if len(entry.value) > 500:
                bloated_fields.append((field, entry))

    if bloated_fields:
        console.print(
            f"[yellow]Found {len(bloated_fields)} bloated fields in durable memory.[/yellow]"
        )
        if yes or typer.confirm("Compress bloated fields?"):
            from ..compression import compress_user_prompt

            for field, entry in bloated_fields:
                old_len = len(entry.value)
                entry.value = compress_user_prompt(entry.value)
                console.print(
                    f"  - {field}: {old_len} -> {len(entry.value)} chars"
                )
            memory_file.write_text(state.to_tok())
            console.print("[green]Durable memory optimized.[/green]")
    else:
        console.print("[green]Durable memory is already optimal.[/green]")


# ---------------------------------------------------------------------------
# Savings / Stats / Replay commands
# ---------------------------------------------------------------------------


@app.command("stats")
@app.command("savings")
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
        typer.Option(
            "--breakdown", help="Show per-type compression breakdown"
        ),
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
    )


@app.command("capture-summary", hidden=True)
def capture_summary(
    session_file: Annotated[
        str, typer.Argument(help="Path to .jsonl capture file")
    ],
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
        rows.append(
            ("Last degradation reason", str(summary["degradation_reason"]))
        )
    if float(summary["savings_pct"]) > 0:
        rows.append(
            ("Session savings", f"{float(summary['savings_pct']):.1f}%")
        )

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


@app.command("capture-review", hidden=True)
def capture_review(
    capture_dir: Annotated[
        str, typer.Argument(help="Directory containing captured session files")
    ],
    verdict: Annotated[
        str | None,
        typer.Option(
            "--verdict", help="Filter to clean, watch, or investigate sessions"
        ),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason", help="Filter sessions by degradation-reason substring"
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit the number of displayed sessions"),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json", help="Write structured review output to a JSON file"
        ),
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
        typer.Option(
            "--stress-dir", help="Optional stress-language artifact directory"
        ),
    ] = None,
    fixtures_dir: Annotated[
        Path,
        typer.Option(
            "--fixtures-dir", help="Replay-fixture metadata directory"
        ),
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
                    str(
                        review["aggregate"]["sessions_with_fallback_activity"]
                    ),
                ),
                (
                    "Reacquisition sessions",
                    str(
                        review["aggregate"][
                            "sessions_with_reacquisition_pressure"
                        ]
                    ),
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
        console.print(
            "[dim]No captured sessions matched the requested filters.[/dim]"
        )

    if candidates and candidate_rows:
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
        console.print(
            "[bold]Coverage candidates:[/bold] "
            + ", ".join(str(item["candidate"]) for item in coverage_rows)
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
        payload = {
            "sessions": review["sessions"],
            "aggregate": review["aggregate"],
            "candidates": candidate_rows,
        }
        if stress_evidence is not None:
            payload["stress_evidence"] = stress_evidence
        if coverage:
            payload["coverage"] = coverage_rows
        json_out.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Wrote capture review:[/green] {json_out}")


@app.command("evidence-gap", hidden=True)
def evidence_gap(
    capture_dir: Annotated[
        str, typer.Argument(help="Directory containing captured session files")
    ],
    stress_dir: Annotated[
        Path | None,
        typer.Option(
            "--stress-dir", help="Optional stress-language artifact directory"
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json", help="Write structured coverage output to JSON"
        ),
    ] = None,
    fixtures_dir: Annotated[
        Path,
        typer.Option(
            "--fixtures-dir", help="Replay-fixture metadata directory"
        ),
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
        json_out.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Wrote evidence gap:[/green] {json_out}")


@app.command(hidden=True)
def replay(
    session_file: Annotated[
        str, typer.Argument(help="Path to .jsonl capture file")
    ],
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


# ---------------------------------------------------------------------------
# Doctor command
# ---------------------------------------------------------------------------


@app.command()
def doctor(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show all behavior signals")
    ] = False,
) -> None:
    """Check bridge health and runtime contract conformance."""
    from ._release import doctor_command

    doctor_command(verbose=verbose)


# ---------------------------------------------------------------------------
# Gate-check / JIT commands
# ---------------------------------------------------------------------------


@app.command("jit-check", hidden=True)
def jit_check() -> None:
    """Run Macro JIT execution verification (autonomous symbolic engine)."""
    console.print("[bold cyan]Running Tok Macro JIT Gate Check...[/bold cyan]")
    try:
        import pytest

        retcode = pytest.main(["tests/unit/test_jit_execution.py", "-v"])
        if retcode != 0:
            console.print("[red]❌ Macro JIT Gate Check failed![/red]")
            raise typer.Exit(1)
        console.print("[green]✅ Macro JIT Gate Check passed![/green]")
    except ImportError:
        console.print(
            "[yellow]⚠️ pytest not found; falling back to unittest discovery...[/yellow]"
        )
        import unittest

        suite = unittest.TestLoader().discover(
            "tests/unit", pattern="test_jit_execution.py"
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            console.print("[red]❌ Macro JIT Gate Check failed![/red]")
            raise typer.Exit(1) from None
        console.print("[green]✅ Macro JIT Gate Check passed![/green]")


@app.command("gate-check", hidden=True)
def gate_check(
    fixtures_dir: Annotated[
        Path, typer.Argument(help="Directory containing replay fixtures")
    ],
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
        typer.Option(
            "--export",
            "-e",
            help="Path to export gate results JSON",
        ),
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
        typer.Option(
            "--set",
            help="Fixture set to use (feature, full, or redteam)",
        ),
    ] = None,
    emit_metrics: Annotated[
        Path | None,
        typer.Option(
            "--emit-metrics",
            help="Alias for --export (baseline_metrics.json)",
        ),
    ] = None,
    stability_dir: Annotated[
        Path | None,
        typer.Option(
            "--stability-dir",
            help="Directory of *_stability.json files from live-benchmark runs. "
            "Checked against --required-benchmarks pass criteria.",
        ),
    ] = None,
    required_benchmarks: Annotated[
        str,
        typer.Option(
            "--required-benchmarks",
            help="Comma-separated list of benchmark names that must be present and "
            "passing in --stability-dir (default: coding-loop-5,research-loop-5).",
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
        required_benchmarks=required_benchmarks,
    )


# ---------------------------------------------------------------------------
# Legacy aliases (hidden)
# ---------------------------------------------------------------------------


@app.command(hidden=True)
def generate_fixture(
    type: Annotated[
        str, typer.Argument(help="Fixture type: coding, search, pressure")
    ],
    name: Annotated[str, typer.Argument(help="Fixture name")],
    template: Annotated[
        str, typer.Option("--template", "-t", help="Metadata template")
    ] = "standard_claude",
    turns: Annotated[
        int,
        typer.Option("--turns", help="Number of turns for coding fixtures"),
    ] = 5,
    searches: Annotated[
        int,
        typer.Option(
            "--searches", help="Number of searches for search fixtures"
        ),
    ] = 8,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats", help="Number of repeats for pressure fixtures"
        ),
    ] = 6,
    complexity: Annotated[
        str,
        typer.Option(
            "--complexity",
            "-c",
            help="Complexity level: simple, medium, complex",
        ),
    ] = "medium",
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory")
    ] = "tests/fixtures/replay",
) -> None:
    """Legacy root alias for `tok dev generate-fixture`."""
    dev_generate_fixture(
        type, name, template, turns, searches, repeats, complexity, output
    )


@app.command("live-benchmark", hidden=True)
def live_benchmark(
    benchmark: Annotated[
        str, typer.Option("--benchmark", help="Benchmark definition to run")
    ] = "coding-loop",
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Run baseline, tok-minimal, tok-native, tok-tool-compatible, or compare",
        ),
    ] = "compare",
    model: Annotated[
        str, typer.Option("--model", help="Model identifier to use")
    ] = "deepseek/deepseek-v3.2",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for artifacts"),
    ] = None,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature")
    ] = 0.0,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Completion token cap")
    ] = 300,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Request timeout in seconds")
    ] = 120.0,
    turns: Annotated[
        int | None,
        typer.Option("--turns", help="Number of benchmark turns to run"),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats", help="Repeat compare mode N times for stability"
        ),
    ] = 1,
    pricing_prompt: Annotated[
        float | None,
        typer.Option(
            "--pricing-prompt", help="Prompt token price per 1M tokens (USD)"
        ),
    ] = None,
    pricing_completion: Annotated[
        float | None,
        typer.Option(
            "--pricing-completion",
            help="Completion token price per 1M tokens (USD)",
        ),
    ] = None,
    provider_options: Annotated[
        str | None,
        typer.Option(
            "--provider-options",
            help="JSON object passed as extra_body to the provider (e.g. OpenRouter routing options)",
        ),
    ] = None,
) -> None:
    """Legacy root alias for `tok dev live-benchmark`."""
    dev_live_benchmark(
        benchmark,
        mode,
        model,
        output,
        temperature,
        max_tokens,
        timeout,
        turns,
        repeats,
        pricing_prompt,
        pricing_completion,
        provider_options,
    )


@app.command("stress-language", hidden=True)
def stress_language(
    model: Annotated[
        str, typer.Option("--model", help="Model identifier to use")
    ] = "qwen/qwen3-coder-next",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for artifacts"),
    ] = None,
    target_breakpoints: Annotated[
        int,
        typer.Option(
            "--target-breakpoints",
            help="Stop after N distinct breakpoint classes",
        ),
    ] = 5,
    max_tasks: Annotated[
        int, typer.Option("--max-tasks", help="Hard cap on task count")
    ] = 24,
    max_tool_rounds: Annotated[
        int,
        typer.Option(
            "--max-tool-rounds", help="Hard cap on tool rounds per task"
        ),
    ] = 8,
    max_retries_per_task: Annotated[
        int,
        typer.Option(
            "--max-retries-per-task",
            help="Retry budget for failed task answers",
        ),
    ] = 2,
    min_payload_pressure_bytes: Annotated[
        int,
        typer.Option(
            "--min-payload-pressure-bytes",
            help="Evidence volume required before payload pressure is considered reached",
        ),
    ] = 12000,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature")
    ] = 0.0,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Completion token cap")
    ] = 450,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress", help="Show live progress logs"
        ),
    ] = True,
    provider_options: Annotated[
        str | None,
        typer.Option(
            "--provider-options",
            help="JSON object passed as extra_body to the provider",
        ),
    ] = None,
    required_classes: Annotated[
        str | None,
        typer.Option(
            "--required-classes",
            help="Comma-separated required breakpoint classes. Use | for alternatives.",
        ),
    ] = None,
) -> None:
    """Legacy root alias for `tok dev stress-language`."""
    dev_stress_language(
        model,
        output,
        target_breakpoints,
        max_tasks,
        max_tool_rounds,
        max_retries_per_task,
        min_payload_pressure_bytes,
        temperature,
        max_tokens,
        progress,
        provider_options,
        required_classes,
    )


@app.command("pressure", hidden=True)
def pressure(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
    export: Annotated[
        str, typer.Option("--export", "-e", help="Export results to file")
    ] = "",
) -> None:
    """Legacy root alias for `tok metrics pressure`."""
    metrics_pressure(window, export)


@app.command("memory", hidden=True)
def memory(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
) -> None:
    """Legacy root alias for `tok metrics memory`."""
    metrics_memory(window)


@app.command("savings-trend", hidden=True)
def savings_trend(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
) -> None:
    """Legacy root alias for `tok metrics savings-trend`."""
    metrics_savings_trend(window)


@app.command("fallback", hidden=True)
def fallback(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
) -> None:
    """Legacy root alias for `tok metrics fallback`."""
    metrics_fallback(window)


@app.command("health", hidden=True)
def health(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
    export: Annotated[
        str, typer.Option("--export", "-e", help="Export results to file")
    ] = "",
) -> None:
    """Legacy root alias for `tok metrics health`."""
    metrics_health(window, export)


if __name__ == "__main__":
    app()
