from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ._cli_support import console
from ._dev import generate_fixture as dev_generate_fixture
from ._dev import live_benchmark as dev_live_benchmark
from ._dev import stress_language as dev_stress_language
from ._metrics import fallback as metrics_fallback
from ._metrics import health as metrics_health
from ._metrics import memory as metrics_memory
from ._metrics import pressure as metrics_pressure
from ._metrics import savings_trend as metrics_savings_trend


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
        console.print("[yellow]⚠️ pytest not found; falling back to unittest discovery...[/yellow]")
        import unittest

        suite = unittest.TestLoader().discover("tests/unit", pattern="test_jit_execution.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            console.print("[red]❌ Macro JIT Gate Check failed![/red]")
            raise typer.Exit(1) from None
        console.print("[green]✅ Macro JIT Gate Check passed![/green]")


def generate_fixture(
    type: Annotated[str, typer.Argument(help="Fixture type: coding, search, pressure")],
    name: Annotated[str, typer.Argument(help="Fixture name")],
    template: Annotated[str, typer.Option("--template", "-t", help="Metadata template")] = "standard_claude",
    turns: Annotated[
        int,
        typer.Option("--turns", help="Number of turns for coding fixtures"),
    ] = 5,
    searches: Annotated[
        int,
        typer.Option("--searches", help="Number of searches for search fixtures"),
    ] = 8,
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Number of repeats for pressure fixtures"),
    ] = 6,
    complexity: Annotated[
        str,
        typer.Option(
            "--complexity",
            "-c",
            help="Complexity level: simple, medium, complex",
        ),
    ] = "medium",
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "tests/fixtures/replay",
) -> None:
    """Legacy root alias for `tok dev generate-fixture`."""
    dev_generate_fixture(type, name, template, turns, searches, repeats, complexity, output)


def live_benchmark(
    benchmark: Annotated[str, typer.Option("--benchmark", help="Benchmark definition to run")] = "coding-loop",
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Run baseline, tok-minimal, tok-native, tok-tool-compatible, or compare",
        ),
    ] = "compare",
    model: Annotated[str, typer.Option("--model", help="Model identifier to use")] = "deepseek/deepseek-v3.2",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for artifacts"),
    ] = None,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature")] = 0.0,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Completion token cap")] = 1024,
    timeout: Annotated[float, typer.Option("--timeout", help="Request timeout in seconds")] = 120.0,
    turns: Annotated[
        int | None,
        typer.Option("--turns", help="Number of benchmark turns to run"),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Repeat compare mode N times for stability"),
    ] = 1,
    pricing_prompt: Annotated[
        float | None,
        typer.Option("--pricing-prompt", help="Prompt token price per 1M tokens (USD)"),
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
        benchmark=benchmark,
        mode=mode,
        model=model,
        output=output,
        program="legacy",
        catalog_root=Path("benchmarks"),
        family="execution_patch,repo_grounding",
        task=None,
        lane="production_claude_lane",
        include_advisory=False,
        public_release_only=False,
        legacy_benchmarks=None,
        local_debug=False,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        turns=turns,
        repeats=repeats,
        pricing_prompt=pricing_prompt,
        pricing_completion=pricing_completion,
        provider_options=provider_options,
    )


def stress_language(
    model: Annotated[str, typer.Option("--model", help="Model identifier to use")] = "qwen/qwen3-coder-next",
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
    max_tasks: Annotated[int, typer.Option("--max-tasks", help="Hard cap on task count")] = 24,
    max_tool_rounds: Annotated[
        int,
        typer.Option("--max-tool-rounds", help="Hard cap on tool rounds per task"),
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
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature")] = 0.0,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Completion token cap")] = 450,
    progress: Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show live progress logs"),
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


def pressure(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
    export: Annotated[str, typer.Option("--export", "-e", help="Export results to file")] = "",
) -> None:
    """Legacy root alias for `tok metrics pressure`."""
    metrics_pressure(window, export)


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


def health(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
    export: Annotated[str, typer.Option("--export", "-e", help="Export results to file")] = "",
) -> None:
    """Legacy root alias for `tok metrics health`."""
    metrics_health(window, export)


def register(app: typer.Typer) -> None:
    app.command("jit-check", hidden=True)(jit_check)
    app.command(hidden=True)(generate_fixture)
    app.command("live-benchmark", hidden=True)(live_benchmark)
    app.command("stress-language", hidden=True)(stress_language)
    app.command("pressure", hidden=True)(pressure)
    app.command("memory", hidden=True)(memory)
    app.command("savings-trend", hidden=True)(savings_trend)
    app.command("fallback", hidden=True)(fallback)
    app.command("health", hidden=True)(health)
