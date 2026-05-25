"""tok benchmark — deterministic compression benchmark (experimental).

Usage:
    tok benchmark run [--fixtures PATH] [--output PATH]
    tok benchmark report [--input PATH] [--markdown] [--json]

Fixtures ship with the package. Results are reproducible across installs:
    tok benchmark run && tok benchmark report
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ._cli_support import console

# Packaged fixtures live alongside the installed source so tok benchmark run
# works from both a wheel install and a source checkout.
_PACKAGED_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "bench_fixtures" / "replay"
# Source-checkout fallback for contributors running against the full test suite.
_SOURCE_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "replay"
_FIXTURE_DIR = _PACKAGED_FIXTURE_DIR if _PACKAGED_FIXTURE_DIR.is_dir() else _SOURCE_FIXTURE_DIR


def _default_results_dir() -> Path:
    """Return a user-writable benchmark report directory."""
    override = os.getenv("TOK_BENCHMARK_RESULTS_DIR")
    if override:
        return Path(override).expanduser()

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "tok"

    return Path.home() / ".cache" / "tok"


_RESULTS_DIR = _default_results_dir()
_LATEST_REPORT = _RESULTS_DIR / "benchmark_latest.json"


@contextmanager
def _quiet_benchmark_logging(enabled: bool):
    """Temporarily silence logging from compression/replay internals."""
    if not enabled:
        yield
        return

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


# ---------------------------------------------------------------------------
# Internal helpers — pure functions, unit-testable without CLI
# ---------------------------------------------------------------------------


def _collect_fixtures(fixture_dir: Path) -> list[Path]:
    """Return sorted list of .jsonl replay fixtures (excludes .meta.json files)."""
    return sorted(p for p in fixture_dir.glob("*.jsonl") if ".meta" not in p.name)


def _fixture_to_report_entry(fixture_path: Path) -> dict:
    """Run analyze_replay_fixture() and return a benchmark schema entry."""
    from tok.utils.replay_metrics import analyze_replay_fixture

    metrics = analyze_replay_fixture(fixture_path)
    meta_path = fixture_path.with_suffix(fixture_path.suffix + ".meta.json")
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    fallback_events = metrics.behavior_totals.get("non_tok_response", 0) + metrics.behavior_totals.get(
        "fail_open_compat_response", 0
    )
    return {
        "input_saved_pct": round(metrics.savings_pct, 2),
        # fallback_rate is derived from raw counts; store both for lossless aggregation
        "fallback_events": fallback_events,
        "fallback_rate": round(min(fallback_events / max(metrics.lines, 1), 1.0), 4),
        "turns": metrics.lines,
        "strategy_breakdown": {k: v for k, v in metrics.type_savings_tokens.items() if v > 0},
        "total_before_tokens": metrics.total_before_tokens,
        "total_after_tokens": metrics.total_after_tokens,
        "input_saved_tokens": metrics.input_saved_tokens,
        "fixture_kind": meta.get("fixture_kind", "unknown"),
    }


def _build_report(fixture_dir: Path) -> dict:
    """Build the full benchmark report dict from a directory of .jsonl fixtures."""
    fixtures = _collect_fixtures(fixture_dir)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for path in fixtures:
        try:
            results[path.stem] = _fixture_to_report_entry(path)
        except Exception as exc:  # noqa: BLE001
            errors[path.stem] = str(exc)

    aggregate: dict = {}
    if results:
        savings_pcts = sorted(v["input_saved_pct"] for v in results.values())
        total_turns = sum(v["turns"] for v in results.values())
        # Aggregate from raw event counts, not reconstructed from rounded per-fixture rates
        total_fallback_events = sum(v["fallback_events"] for v in results.values())
        total_saved = sum(v["input_saved_tokens"] for v in results.values())
        # Use nearest-rank p90: index = ceil(0.9 * n) - 1
        p90_idx = max(0, -(-len(savings_pcts) * 9 // 10) - 1)
        aggregate = {
            "median_saved_pct": round(statistics.median(savings_pcts), 2),
            "p90_saved_pct": round(savings_pcts[p90_idx], 2),
            "total_tokens_saved": total_saved,
            # Weighted by turns so a 1-turn fixture doesn't equal a 100-turn fixture
            "fallback_rate": round(total_fallback_events / max(total_turns, 1), 4),
            "fixture_count": len(results),
            "error_count": len(errors),
        }

    return {
        "schema_version": "1",
        "date": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "fixtures": results,
        "aggregate": aggregate,
        "errors": errors,
    }


def _build_report_with_optional_quiet(fixture_dir: Path, *, quiet: bool) -> dict:
    with _quiet_benchmark_logging(quiet):
        return _build_report(fixture_dir)


def _top_strategy(entry: dict) -> str:
    """Return the name of the strategy with the highest token savings."""
    breakdown = entry.get("strategy_breakdown", {})
    if not breakdown:
        return "—"
    top_strategy = max(breakdown, key=lambda k: breakdown[k])
    return str(top_strategy)


def _render_rich_table(report: dict) -> None:
    """Print a Rich table + aggregate summary to the console."""
    fixtures = report.get("fixtures", {})
    aggregate = report.get("aggregate", {})

    table = Table(title="Tok Compression Benchmark", show_lines=False)
    table.add_column("Fixture", style="cyan", no_wrap=True)
    table.add_column("Turns", justify="right")
    table.add_column("Before (tok)", justify="right")
    table.add_column("After (tok)", justify="right")
    table.add_column("Saved %", justify="right", style="green")
    table.add_column("Fallback", justify="right")
    table.add_column("Top Strategy")

    for name, entry in sorted(fixtures.items()):
        table.add_row(
            name,
            str(entry["turns"]),
            f"{entry['total_before_tokens']:,}",
            f"{entry['total_after_tokens']:,}",
            f"{entry['input_saved_pct']:.1f}%",
            f"{entry['fallback_rate']:.1%}",
            _top_strategy(entry),
        )

    console.print(table)

    if aggregate:
        console.print()
        console.print(
            f"[bold]Aggregate:[/bold] median {aggregate['median_saved_pct']}% saved"
            f" · p90 {aggregate['p90_saved_pct']}%"
            f" · {aggregate['total_tokens_saved']:,} tokens saved"
            f" · fallback {aggregate['fallback_rate']:.1%}"
            f" · {aggregate['fixture_count']} fixtures"
        )


def _render_markdown(report: dict) -> str:
    """Return BENCHMARK.md markdown string from a report dict."""
    fixtures = report.get("fixtures", {})
    aggregate = report.get("aggregate", {})
    date = report.get("date", "unknown")[:10]
    python_version = report.get("python_version", "unknown")

    lines = [
        "# Tok Compression Benchmark (Experimental)",
        "",
        f"> Last run: {date} | Python {python_version} | Status: experimental",
        "> Fixtures ship with the package. Reproduce from any install: `tok benchmark run && tok benchmark report`",
        "> Note: fixtures are designed by the Tok maintainers — treat results as internal regression data, not independent proof.",
        "",
    ]

    if aggregate:
        lines += [
            "## Aggregate Results",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Median token savings | {aggregate.get('median_saved_pct', 0):.1f}% |",
            f"| P90 token savings | {aggregate.get('p90_saved_pct', 0):.1f}% |",
            f"| Total tokens saved (all fixtures) | {aggregate.get('total_tokens_saved', 0):,} |",
            f"| Average fallback rate | {aggregate.get('fallback_rate', 0):.1%} |",
            f"| Fixtures run | {aggregate.get('fixture_count', 0)} |",
            "",
        ]
    else:
        lines += ["## Aggregate Results", "", "_No fixtures processed._", ""]

    lines += [
        "## Per-Fixture Results",
        "",
        "| Fixture | Turns | Saved % | Fallback Rate | Top Strategy |",
        "|---|---|---|---|---|",
    ]
    for name, entry in sorted(fixtures.items()):
        lines.append(
            f"| {name} | {entry['turns']} | {entry['input_saved_pct']:.1f}% "
            f"| {entry['fallback_rate']:.1%} | {_top_strategy(entry)} |"
        )

    lines += [
        "",
        "## Methodology",
        "",
        "- Fixtures: deterministic JSONL replay captures bundled at `tok/bench_fixtures/replay/`",
        "- Token counting: tiktoken cl100k\\_base (fallback: chars/4)",
        "- Compression pipeline: `tok.utils.replay_metrics.analyze_replay_fixture()`",
        "- Fallback rate: `min(1, (non_tok_response + fail_open_compat_response) / turns)` — clamped to 1.0",
        "- Note: some fixtures are specifically designed to test fallback behavior, so high fallback rates are expected for those",
        "- Default results file: user cache path `tok/benchmark_latest.json`",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    benchmark_app = typer.Typer(help="Deterministic compression benchmark (experimental)", no_args_is_help=True)
    app.add_typer(benchmark_app, name="benchmark", hidden=True)

    @benchmark_app.command("run")
    def benchmark_run(
        fixture_dir: Annotated[
            Path,
            typer.Option("--fixtures", help="Directory of .jsonl replay fixtures"),
        ] = _FIXTURE_DIR,
        output: Annotated[
            Path,
            typer.Option("--output", help="Output JSON report file"),
        ] = _LATEST_REPORT,
        quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
        allow_errors: Annotated[
            bool,
            typer.Option("--allow-errors", help="Exit 0 even if some fixtures fail to process"),
        ] = False,
    ) -> None:
        """Run the full fixture suite and write results/benchmark_latest.json.

        Exits non-zero if any fixture errors occur, unless --allow-errors is passed.
        """
        if not fixture_dir.is_dir():
            console.print(f"[red]Fixture directory not found: {fixture_dir}[/red]")
            console.print(
                "Pass [cyan]--fixtures PATH[/cyan] to a directory of .jsonl replay fixtures,\n"
                "or reinstall tok-protocol (the bundled fixtures may be missing from your install)."
            )
            raise typer.Exit(1)

        if not quiet:
            console.print(f"[bold]Running benchmark[/bold] against {fixture_dir} …")

        report = _build_report_with_optional_quiet(fixture_dir, quiet=quiet)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")

        err = len(report["errors"])
        if not quiet:
            n = len(report["fixtures"])
            agg = report.get("aggregate", {})
            med = agg.get("median_saved_pct", 0)
            console.print(
                f"[green]Done.[/green] {n} fixtures · median {med:.1f}% saved"
                + (f" · [red]{err} errors[/red]" if err else "")
            )
            console.print(f"Report written to [cyan]{output}[/cyan]")

        if err and not allow_errors:
            raise typer.Exit(1)

    @benchmark_app.command("report")
    def benchmark_report(
        input_file: Annotated[
            Path,
            typer.Option("--input", help="JSON report file to render"),
        ] = _LATEST_REPORT,
        markdown: Annotated[bool, typer.Option("--markdown", help="Render as Markdown")] = False,
        json_out: Annotated[bool, typer.Option("--json", help="Print raw JSON")] = False,
    ) -> None:
        """Render benchmark results as Rich table, Markdown, or JSON."""
        if not input_file.exists():
            console.print(f"[red]Report file not found: {input_file}[/red]")
            console.print("Run [cyan]tok benchmark run[/cyan] first.")
            raise typer.Exit(1)

        report = json.loads(input_file.read_text())

        if json_out:
            console.print(json.dumps(report, indent=2))
        elif markdown:
            console.print(_render_markdown(report), end="")
        else:
            _render_rich_table(report)
