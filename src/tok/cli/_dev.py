"""Fixture generation and benchmarking commands for the Tok CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from ._cli_support import console

dev_app = typer.Typer(help="Fixture generation and benchmarking commands")


def _run_legacy_compare_benchmark(
    *,
    benchmark: str,
    runner: Any,
    output: Path,
    turns: int | None,
    repeats: int,
) -> None:
    from tok.testing.live_benchmark import (
        compare_results,
        load_benchmark_definition,
        render_comparison_markdown,
        render_stability_markdown,
        select_preferred_mode,
        summarize_compare_runs,
        summarize_compare_triage,
        write_result,
    )

    definition = load_benchmark_definition(benchmark)
    effective_turns = turns if turns is not None else definition.default_turns
    repeated_results: list[dict[str, Any]] = []
    compare_modes = ("tok-universal",)
    for run_index in range(max(1, repeats)):
        console.print(f"[dim]Running mode: baseline (repeat {run_index + 1})...[/dim]")
        baseline = runner.run(definition, mode="baseline", turns=effective_turns)
        run_results: dict[str, Any] = {"baseline": baseline}
        for compare_mode in compare_modes:
            console.print(f"[dim]Running mode: {compare_mode} (repeat {run_index + 1})...[/dim]")
            run_results[compare_mode] = runner.run(definition, mode=compare_mode, turns=effective_turns)
        repeated_results.append(run_results)
        write_result(output / f"{benchmark}_run{run_index + 1}_baseline.json", run_results["baseline"])
        for compare_mode in compare_modes:
            candidate = run_results[compare_mode]
            write_result(output / f"{benchmark}_run{run_index + 1}_{compare_mode}.json", candidate)
            write_result(
                output / f"{benchmark}_run{run_index + 1}_compare_{compare_mode}.json",
                compare_results(run_results["baseline"], candidate),
            )

    last_run = repeated_results[-1]
    baseline = last_run["baseline"]
    comparisons = [compare_results(baseline, last_run[compare_mode]) for compare_mode in compare_modes]
    preferred_mode = select_preferred_mode(baseline, comparisons)
    write_result(output / f"{benchmark}_baseline.json", baseline)
    for compare_mode in compare_modes:
        compare_result = last_run[compare_mode]
        write_result(output / f"{benchmark}_{compare_mode}.json", compare_result)
        write_result(
            output / f"{benchmark}_compare_{compare_mode}.json",
            compare_results(baseline, compare_result),
        )
    (output / f"{benchmark}_compare.md").write_text(
        render_comparison_markdown(
            baseline,
            comparisons,
        )
    )
    if repeats > 1:
        stability_summary = summarize_compare_runs(repeated_results)
        (output / f"{benchmark}_stability.json").write_text(json.dumps(stability_summary, indent=2))
        (output / f"{benchmark}_stability.md").write_text(
            render_stability_markdown(benchmark, str(getattr(runner, "model", "")), stability_summary)
        )
    triage_summary = summarize_compare_triage(repeated_results)
    (output / f"{benchmark}_triage.json").write_text(json.dumps(triage_summary, indent=2))
    console.print(
        f"[green]✅ Live benchmark complete:[/green] {benchmark} "
        f"baseline_tokens={baseline.provider_usage.total_tokens} "
        f"preferred_tokens={last_run[preferred_mode].provider_usage.total_tokens if preferred_mode in last_run else baseline.provider_usage.total_tokens}"
    )
    console.print(f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_baseline.json'}")
    for compare_mode in compare_modes:
        console.print(f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_{compare_mode}.json'}")
    console.print(f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_compare.md'}")
    console.print(f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_triage.json'}")
    if repeats > 1:
        console.print(f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_stability.json'}")
        console.print(f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_stability.md'}")
    console.print(f"[cyan]Best mode:[/cyan] {preferred_mode}")


def _run_legacy_compare_suite(
    *,
    benchmarks: tuple[str, ...],
    runner: Any,
    output: Path,
    turns: int | None,
    repeats: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for benchmark in benchmarks:
        _run_legacy_compare_benchmark(
            benchmark=benchmark,
            runner=runner,
            output=output,
            turns=turns,
            repeats=repeats,
        )


@dev_app.command("generate-fixture")
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
    """Generate replay fixtures for testing."""
    from tok.testing.fixture_generator import FixtureGenerator

    generator = FixtureGenerator()

    if type == "coding":
        fixture, metadata = generator.generate_coding_session(name, turns, template, complexity)
    elif type == "search":
        fixture, metadata = generator.generate_search_session(name, searches, template)
    elif type == "pressure":
        fixture, metadata = generator.generate_high_pressure_session(name, repeats, template)
    else:
        console.print(f"[red]Unknown fixture type: {type}[/red]")
        console.print("Valid types: coding, search, pressure")
        raise typer.Exit(1)

    generator.save_fixture(name, fixture, metadata, output)
    console.print(f"[green]✅ Generated {type} fixture: {name}[/green]")


@dev_app.command("live-benchmark")
def live_benchmark(
    benchmark: Annotated[str, typer.Option("--benchmark", help="Benchmark definition to run")] = "coding-loop",
    program: Annotated[
        str,
        typer.Option(
            "--program",
            help="Run the legacy replay harness, the catalog executor, or both",
        ),
    ] = "legacy",
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Run baseline, tok-universal, or compare",
        ),
    ] = "compare",
    model: Annotated[str, typer.Option("--model", help="Model identifier to use")] = "deepseek/deepseek-v3.2",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for artifacts"),
    ] = None,
    catalog_root: Annotated[
        Path,
        typer.Option("--catalog-root", help="Benchmark catalog root"),
    ] = Path("benchmarks"),
    family: Annotated[
        str,
        typer.Option(
            "--family",
            help="Comma-separated catalog families to run",
        ),
    ] = "execution_patch,repo_grounding",
    task: Annotated[
        list[str] | None,
        typer.Option("--task", help="Specific catalog task id to run"),
    ] = None,
    lane: Annotated[
        str,
        typer.Option("--lane", help="Benchmark lane id for catalog runs"),
    ] = "production_claude_lane",
    include_advisory: Annotated[
        bool,
        typer.Option("--include-advisory", help="Include advisory real-session episodes"),
    ] = False,
    public_release_only: Annotated[
        bool,
        typer.Option("--public-release-only", help="Run only tasks with public_release=true"),
    ] = False,
    legacy_benchmarks: Annotated[
        str | None,
        typer.Option(
            "--legacy-benchmarks",
            help="Comma-separated replay benchmarks to run when --program both is used",
        ),
    ] = None,
    local_debug: Annotated[
        bool,
        typer.Option(
            "--local-debug",
            help="Allow non-reportable local catalog runs against a dirty checkout",
        ),
    ] = False,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature")] = 0.0,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Completion token cap")] = 300,
    timeout: Annotated[float, typer.Option("--timeout", help="Request timeout in seconds")] = 120.0,
    turns: Annotated[
        int | None,
        typer.Option("--turns", help="Number of benchmark turns to run"),
    ] = None,
    repeats: Annotated[
        int | None,
        typer.Option("--repeats", help="Repeat compare mode N times for stability"),
    ] = None,
    private_evaluator_root: Annotated[
        Path | None,
        typer.Option(
            "--private-evaluator-root",
            help="Path to a private evaluator overlay for claimable execution_patch tasks",
        ),
    ] = None,
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
    """Run a controlled live benchmark in baseline, Tok, or compare mode."""
    from tok.testing.benchmark_executor import (
        render_combined_benchmark_summary,
        run_catalog_benchmark_suite,
    )
    from tok.testing.benchmark_suite import load_benchmark_catalog, render_benchmark_report_markdown
    from tok.testing.live_benchmark import (
        LiveBenchmarkRunner,
        load_benchmark_definition,
        write_result,
    )

    pricing: dict[str, float] | None = None
    if pricing_prompt is not None or pricing_completion is not None:
        pricing = {
            "prompt": pricing_prompt or 0.0,
            "completion": pricing_completion or 0.0,
        }
    parsed_provider_options: dict[str, Any] | None = None
    if provider_options:
        import json as _json

        try:
            parsed_provider_options = _json.loads(provider_options)
        except Exception as exc:
            msg = f"--provider-options must be valid JSON: {exc}"
            raise typer.BadParameter(msg) from exc
    runner = LiveBenchmarkRunner(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        pricing=pricing,
        provider_options=parsed_provider_options,
    )

    if program not in {"legacy", "catalog", "both"}:
        msg = f"--program must be one of legacy, catalog, both (got {program})"
        raise typer.BadParameter(msg)

    if output is None:
        output = Path.cwd() / "tmp" / "live_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    effective_repeats = repeats if repeats is not None else (3 if program in {"catalog", "both"} else 1)

    if program in {"catalog", "both"}:
        if mode != "compare":
            msg = "catalog benchmarks only support --mode compare"
            raise typer.BadParameter(msg)
        catalog = load_benchmark_catalog(catalog_root)
        family_names = tuple(value.strip() for value in family.split(",") if value.strip())
        task_ids = tuple(task or ())
        catalog_output = output / "catalog" if program == "both" else output
        catalog_run = run_catalog_benchmark_suite(
            catalog=catalog,
            lane_id=lane,
            output_root=catalog_output,
            repeats=max(1, effective_repeats),
            families=family_names,
            task_ids=task_ids,
            include_advisory=include_advisory,
            public_release_only=public_release_only,
            local_debug=local_debug,
            runner=runner,
            repo_root=Path.cwd(),
            private_evaluator_root=private_evaluator_root,
        )
        catalog_markdown = render_benchmark_report_markdown(catalog_run.report)
        (catalog_output / "report.md").write_text(catalog_markdown)
        console.print(f"[green]✅ Catalog benchmark complete:[/green] {catalog_output / 'report.json'}")
        console.print(f"[cyan]Markdown:[/cyan] {catalog_output / 'report.md'}")
        if program == "catalog":
            return

        selected_legacy = tuple(
            value.strip()
            for value in (legacy_benchmarks or "coding-loop-5,research-loop-5").split(",")
            if value.strip()
        )
        legacy_output = output / "legacy"
        _run_legacy_compare_suite(
            benchmarks=selected_legacy,
            runner=runner,
            output=legacy_output,
            turns=turns,
            repeats=effective_repeats,
        )
        summary = render_combined_benchmark_summary(
            legacy_benchmarks=selected_legacy,
            catalog_run=catalog_run,
            catalog_report_markdown=catalog_markdown,
        )
        (output / "summary.md").write_text(summary)
        console.print(f"[green]✅ Combined benchmark summary:[/green] {output / 'summary.md'}")
        return

    definition = load_benchmark_definition(benchmark)
    effective_turns = turns if turns is not None else definition.default_turns

    if mode == "compare":
        _run_legacy_compare_benchmark(
            benchmark=benchmark,
            runner=runner,
            output=output,
            turns=effective_turns,
            repeats=effective_repeats,
        )
        return

    if mode not in {
        "baseline",
        "tok-universal",
    }:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        raise typer.Exit(1)

    result = runner.run(definition, mode=mode, turns=effective_turns)
    write_result(output / f"{benchmark}_{mode}.json", result)
    console.print(
        f"[green]✅ Live benchmark complete:[/green] {benchmark} mode={mode} "
        f"tokens={result.provider_usage.total_tokens} turns={result.turn_count} "
        f"success={result.task_success}"
    )
    console.print(f"[cyan]Artifact:[/cyan] {output / f'{benchmark}_{mode}.json'}")


@dev_app.command("compression-frontier")
def compression_frontier(
    model: Annotated[str, typer.Option("--model", help="Model identifier to use")] = "deepseek/deepseek-v3.2",
    benchmarks: Annotated[
        str,
        typer.Option(
            "--benchmarks",
            help="Comma-separated benchmark names to include",
        ),
    ] = "coding-loop-5,research-loop-5,research-loop-8",
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="How many repeated runs to execute per rung"),
    ] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for artifacts"),
    ] = None,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature")] = 0.0,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Completion token cap")] = 300,
    timeout: Annotated[float, typer.Option("--timeout", help="Request timeout in seconds")] = 120.0,
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
            help="JSON object passed as extra_body to the provider",
        ),
    ] = None,
    openrouter_prompt: Annotated[
        str,
        typer.Option(
            "--openrouter-prompt",
            help="Prompt used for the cheap OpenRouter probe loop",
        ),
    ] = "Give me a one-line repo summary.",
    openrouter_turns: Annotated[
        str,
        typer.Option(
            "--openrouter-turns",
            help="Comma-separated turn counts for the cheap OpenRouter probe",
        ),
    ] = "5,12",
    openrouter_delay: Annotated[
        float,
        typer.Option(
            "--openrouter-delay",
            help="Delay between OpenRouter probe turns in seconds",
        ),
    ] = 0.2,
    baseline_ref: Annotated[
        str,
        typer.Option(
            "--baseline-ref",
            help="Checkpoint to treat as the calmer pre-natural-first baseline",
        ),
    ] = "5aebb5d",
    current_only: Annotated[
        bool,
        typer.Option(
            "--current-only",
            help="Only run the current checkout and skip exported checkpoint comparisons",
        ),
    ] = False,
) -> None:
    """Find the highest compression rung that still stays calm."""
    from tok.testing.frontier import (
        DEFAULT_FRONTIER_PROFILES,
        FrontierCheckpoint,
        render_frontier_markdown,
        run_frontier_report,
        select_frontier_checkpoints,
    )

    parsed_provider_options: dict[str, Any] | None = None
    if provider_options:
        try:
            parsed_provider_options = json.loads(provider_options)
        except Exception as exc:
            msg = f"--provider-options must be valid JSON: {exc}"
            raise typer.BadParameter(msg) from exc

    pricing: dict[str, float] | None = None
    if pricing_prompt is not None or pricing_completion is not None:
        pricing = {
            "prompt": pricing_prompt or 0.0,
            "completion": pricing_completion or 0.0,
        }

    repo_root = Path.cwd()
    selected_checkpoints = (
        [FrontierCheckpoint(label="current-head", ref="CURRENT")]
        if current_only
        else select_frontier_checkpoints(repo_root, baseline_ref=baseline_ref)
    )
    benchmark_list = [value.strip() for value in benchmarks.split(",") if value.strip()]
    openrouter_turn_list = [int(value.strip()) for value in openrouter_turns.split(",") if value.strip()]

    if output is None:
        output = repo_root / "tmp" / "compression_frontier"
    output.mkdir(parents=True, exist_ok=True)

    report = run_frontier_report(
        repo_root=repo_root,
        checkpoints=selected_checkpoints,
        profiles=list(DEFAULT_FRONTIER_PROFILES),
        benchmarks=benchmark_list,
        model=model,
        repeats=repeats,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        provider_options=parsed_provider_options,
        pricing=pricing,
        openrouter_prompt=openrouter_prompt,
        openrouter_turn_sets=openrouter_turn_list,
        openrouter_delay_seconds=openrouter_delay,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
    )

    json_path = output / "compression_frontier_report.json"
    md_path = output / "compression_frontier_report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    md_path.write_text(render_frontier_markdown(report))

    console.print(f"[green]✅ Compression frontier complete:[/green] {json_path}")
    console.print(f"[cyan]Markdown:[/cyan] {md_path}")


@dev_app.command("benchmark-validate")
def benchmark_validate(
    root: Annotated[
        Path,
        typer.Option("--root", help="Benchmark catalog root"),
    ] = Path("benchmarks"),
) -> None:
    """Validate the structured production benchmark catalog."""
    from tok.testing.benchmark_suite import load_benchmark_catalog

    catalog = load_benchmark_catalog(root)
    family_counts = catalog.family_counts()

    console.print(
        "[green]✅ Benchmark catalog valid:[/green] "
        f"headline={catalog.headline_lane().id} "
        f"compatibility_lanes={len(catalog.compatibility_lanes())}"
    )
    for family in ("execution_patch", "repo_grounding", "real_session"):
        console.print(f"[cyan]{family}:[/cyan] {family_counts.get(family, 0)} tasks")


@dev_app.command("benchmark-report")
def benchmark_report(
    input: Annotated[
        Path,
        typer.Option("--input", help="JSON report payload to render"),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", help="Benchmark catalog root for raw run payloads"),
    ] = Path("benchmarks"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional markdown output path"),
    ] = None,
) -> None:
    """Render a production benchmark markdown report from JSON."""
    from tok.testing.benchmark_suite import (
        BenchmarkComparisonRun,
        build_benchmark_report,
        load_benchmark_catalog,
        load_benchmark_report,
        render_benchmark_report_markdown,
    )

    payload = json.loads(input.read_text())
    if "lane_summaries" in payload:
        report = load_benchmark_report(input)
    else:
        catalog = load_benchmark_catalog(root)
        runs = [BenchmarkComparisonRun.from_dict(item) for item in payload.get("runs", []) if isinstance(item, dict)]
        if not runs:
            msg = "raw benchmark report payload must contain a non-empty 'runs' list"
            raise typer.BadParameter(msg)
        report = build_benchmark_report(
            catalog,
            runs,
            title=str(payload.get("title") or "Production Tok Benchmark Report"),
        )

    markdown = render_benchmark_report_markdown(report)
    if output is not None:
        output.write_text(markdown)
        console.print(f"[green]✅ Benchmark report written:[/green] {output}")
        return

    console.print(markdown)


@dev_app.command("stress-language")
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
    """Run a long-lived Tok language stress harness against the OpenRouter path."""
    from tok.testing.stress import (
        StressHarness,
        StressHarnessConfig,
        default_output_dir,
        required_class_coverage,
        summarize_implicated_files,
        write_stress_artifacts,
    )

    parsed_provider_options: dict[str, Any] | None = None
    if provider_options:
        try:
            parsed_provider_options = json.loads(provider_options)
        except Exception as exc:
            msg = f"--provider-options must be valid JSON: {exc}"
            raise typer.BadParameter(msg) from exc

    output_dir = output or default_output_dir()
    parsed_required_classes = (
        tuple(item.strip() for item in required_classes.split(",") if item.strip()) if required_classes else None
    )
    config = StressHarnessConfig(
        model=model,
        target_breakpoints=target_breakpoints,
        max_tasks=max_tasks,
        max_tool_rounds=max_tool_rounds,
        max_retries_per_task=max_retries_per_task,
        min_payload_pressure_bytes=min_payload_pressure_bytes,
        temperature=temperature,
        max_tokens=max_tokens,
        provider_options=parsed_provider_options,
        output_dir=output_dir,
        progress=progress,
        required_classes=parsed_required_classes or StressHarnessConfig().required_classes,
    )
    harness = StressHarness(config)
    result = harness.run()
    session = getattr(harness, "session", None)
    if session is None:
        artifacts = write_stress_artifacts(output_dir, result)
    else:
        artifacts = write_stress_artifacts(output_dir, result, session=session)
    classes_seen = sorted({bp.breakpoint_class for bp in result.breakpoints})
    coverage = required_class_coverage(classes_seen, config.required_classes)
    early_retention_probe_ran = any(
        getattr(turn, "phase_name", "") == "retention-probe" and "retention_probe_early" in getattr(turn, "task_id", "")
        for turn in result.turns
    )
    late_retention_probe_ran = any(
        getattr(turn, "phase_name", "") == "retention-probe"
        and "retention_probe_early" not in getattr(turn, "task_id", "")
        for turn in result.turns
    )
    console.print(
        f"[green]✅ Stress harness complete:[/green] model={model} "
        f"tasks={result.tasks_completed} breakpoints={len(result.breakpoints)} "
        f"baseline_only={result.baseline_only}"
    )
    console.print(f"[cyan]Breakpoint classes:[/cyan] {', '.join(classes_seen) if classes_seen else 'none'}")
    console.print(
        f"[cyan]Coverage:[/cyan] complete={coverage['complete']} "
        f"covered={', '.join(coverage['covered']) or 'none'} "
        f"missing={', '.join(coverage['missing']) or 'none'}"
    )
    console.print(
        f"[cyan]Anchors:[/cyan] validated={result.validated_anchor_count} "
        f"before-baseline={result.anchors_before_baseline} "
        f"tool-backed-turns={result.tool_backed_turns}/{len(result.turns)}"
    )
    console.print(
        f"[cyan]Seed phase:[/cyan] searches={result.seed_searches} "
        f"direct-reads={result.seed_direct_reads} "
        f"answer-attempts={result.seed_answer_attempts} "
        f"evidence-sufficient={result.seed_evidence_sufficient}"
    )
    console.print(
        f"[cyan]Memory checks:[/cyan] reuse={result.reuse_checks_run} checkpoint={result.checkpoint_checks_run}"
    )
    console.print(
        f"[cyan]Reuse probes:[/cyan] attempts={result.reuse_probe_attempts} "
        f"successes={result.reuse_probe_successes} "
        f"reacquisitions={result.reacquisition_events_seen}"
    )
    console.print(
        f"[cyan]Retention probes:[/cyan] attempts={result.retention_probe_attempts} "
        f"successes={result.retention_probe_successes} "
        f"substitutions={result.retention_substitution_events_seen}"
    )
    console.print(
        f"[cyan]Tool contract probes:[/cyan] attempts={result.tool_contract_probe_attempts} "
        f"failure_events={result.tool_contract_failure_events_seen}"
    )
    console.print(
        f"[cyan]Tool contract signals:[/cyan] mixed={result.mixed_answer_tool_events_seen} "
        f"unsupported={result.unsupported_tool_events_seen} "
        f"bad_args={result.bad_tool_args_events_seen} "
        f"toolless_fresh={result.toolless_fresh_answer_events_seen}"
    )
    console.print(
        f"[cyan]Late retention probes:[/cyan] attempts={result.late_retention_probe_attempts} "
        f"successes={result.late_retention_probe_successes}"
    )
    console.print(f"[cyan]Early retention probe:[/cyan] {early_retention_probe_ran}")
    console.print(f"[cyan]Late retention probe:[/cyan] {late_retention_probe_ran}")
    console.print(
        f"[cyan]Resend modes:[/cyan] {', '.join(result.resend_modes_seen) or 'none'} "
        f"payload_pressure={result.payload_pressure_reached}"
    )
    console.print(f"[cyan]Compaction eligibility:[/cyan] {result.compaction_eligible}")
    console.print(
        f"[cyan]Run diagnosis:[/cyan] {result.run_diagnosis} "
        f"weak_reasons={', '.join(result.weak_run_reasons) or 'none'}"
    )
    console.print(f"[cyan]First-anchor failure mode:[/cyan] {result.first_anchor_failure_mode}")
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['stress_run']}")
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['breakpoints']}")
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['stress_report']}")
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['language_refactor_plan']}")
    implicated = summarize_implicated_files(result.breakpoints)
    if implicated:
        console.print("[cyan]Implicated files:[/cyan]")
        for item in implicated[:10]:
            console.print(f"  - {item['path']} ({item['count']} mentions)")


@dev_app.command("dedup-frontier")
def dedup_frontier(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for investigation artifacts",
        ),
    ] = Path("tmp/dedup_frontier"),
    fixtures_dir: Annotated[
        Path,
        typer.Option(
            "--fixtures-dir",
            help="Replay fixture directory to analyze",
        ),
    ] = Path("tests/fixtures/replay"),
    fixture: Annotated[
        list[Path] | None,
        typer.Option(
            "--fixture",
            help="Additional explicit replay fixture path(s)",
        ),
    ] = None,
    stress_run: Annotated[
        list[Path] | None,
        typer.Option(
            "--stress-run",
            help="Stress harness artifact JSON path(s) to fold into the report",
        ),
    ] = None,
    bridge_log: Annotated[
        list[Path] | None,
        typer.Option(
            "--bridge-log",
            help="Representative bridge log path(s) such as tokviz.txt",
        ),
    ] = None,
) -> None:
    """Run the replay-first dedup frontier investigation."""
    from tok.analysis import run_dedup_frontier

    artifacts = run_dedup_frontier(
        output_dir=output,
        fixtures_dir=fixtures_dir,
        fixture_paths=list(fixture or []),
        stress_run_paths=list(stress_run or []),
        bridge_log_paths=list(bridge_log or []),
        workspace_root=Path.cwd(),
    )
    console.print("[green]✅ Dedup frontier investigation complete.[/green]")
    console.print(f"[cyan]Artifact:[/cyan] {artifacts['ledger']}")
    console.print(f"[cyan]Artifact:[/cyan] {artifacts['summary']}")
    console.print(f"[cyan]Artifact:[/cyan] {artifacts['report']}")
