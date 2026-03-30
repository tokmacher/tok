from __future__ import annotations

"""Fixture generation and benchmarking commands for the Tok CLI."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ._shared import console

dev_app = typer.Typer(help="Fixture generation and benchmarking commands")


@dev_app.command("generate-fixture")
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
    """Generate replay fixtures for testing."""
    from ..fixture_generator import FixtureGenerator

    generator = FixtureGenerator()

    if type == "coding":
        fixture, metadata = generator.generate_coding_session(
            name, turns, template, complexity
        )
    elif type == "search":
        fixture, metadata = generator.generate_search_session(
            name, searches, template
        )
    elif type == "pressure":
        fixture, metadata = generator.generate_high_pressure_session(
            name, repeats, template
        )
    else:
        console.print(f"[red]Unknown fixture type: {type}[/red]")
        console.print("Valid types: coding, search, pressure")
        raise typer.Exit(1)

    generator.save_fixture(name, fixture, metadata, output)
    console.print(f"[green]✅ Generated {type} fixture: {name}[/green]")


@dev_app.command("live-benchmark")
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
    """Run a controlled live benchmark in baseline, Tok, or compare mode."""
    from ..live_benchmark import (
        LiveBenchmarkRunner,
        compare_results,
        load_benchmark_definition,
        render_comparison_markdown,
        render_stability_markdown,
        select_preferred_mode,
        summarize_compare_runs,
        write_result,
    )

    definition = load_benchmark_definition(benchmark)
    effective_turns = turns if turns is not None else definition.default_turns
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
            raise typer.BadParameter(
                f"--provider-options must be valid JSON: {exc}"
            ) from exc
    runner = LiveBenchmarkRunner(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        pricing=pricing,
        provider_options=parsed_provider_options,
    )

    if output is None:
        output = Path.cwd() / "tmp" / "live_benchmark"
    output.mkdir(parents=True, exist_ok=True)

    if mode == "compare":
        repeated_results: list[dict[str, Any]] = []
        for _ in range(max(1, repeats)):
            console.print(
                f"[dim]Running mode: baseline (repeat {_ + 1})...[/dim]"
            )
            baseline = runner.run(
                definition, mode="baseline", turns=effective_turns
            )
            console.print(
                f"[dim]Running mode: tok-minimal (repeat {_ + 1})...[/dim]"
            )
            tok_minimal = runner.run(
                definition, mode="tok-minimal", turns=effective_turns
            )
            console.print(
                f"[dim]Running mode: tok-native (repeat {_ + 1})...[/dim]"
            )
            tok_native = runner.run(
                definition, mode="tok-native", turns=effective_turns
            )
            console.print(
                f"[dim]Running mode: tok-tool-compatible (repeat {_ + 1})...[/dim]"
            )
            tok_tool_compatible = runner.run(
                definition, mode="tok-tool-compatible", turns=effective_turns
            )
            console.print(
                f"[dim]Running mode: tok-neuro (repeat {_ + 1})...[/dim]"
            )
            tok_neuro = runner.run(
                definition, mode="tok-neuro", turns=effective_turns
            )
            repeated_results.append(
                {
                    "baseline": baseline,
                    "tok-minimal": tok_minimal,
                    "tok-native": tok_native,
                    "tok-tool-compatible": tok_tool_compatible,
                    "tok-neuro": tok_neuro,
                }
            )

        last_run = repeated_results[-1]
        baseline = last_run["baseline"]
        tok_minimal = last_run["tok-minimal"]
        tok_native = last_run["tok-native"]
        tok_tool_compatible = last_run["tok-tool-compatible"]
        tok_neuro = last_run["tok-neuro"]
        minimal_comparison = compare_results(baseline, tok_minimal)
        native_comparison = compare_results(baseline, tok_native)
        tool_compatible_comparison = compare_results(
            baseline, tok_tool_compatible
        )
        neuro_comparison = compare_results(baseline, tok_neuro)
        preferred_mode = select_preferred_mode(
            baseline,
            [
                minimal_comparison,
                native_comparison,
                tool_compatible_comparison,
                neuro_comparison,
            ],
        )
        write_result(output / f"{benchmark}_baseline.json", baseline)
        write_result(output / f"{benchmark}_tok-minimal.json", tok_minimal)
        write_result(output / f"{benchmark}_tok-native.json", tok_native)
        write_result(
            output / f"{benchmark}_tok-tool-compatible.json",
            tok_tool_compatible,
        )
        write_result(output / f"{benchmark}_tok-neuro.json", tok_neuro)
        write_result(
            output / f"{benchmark}_compare_tok-minimal.json",
            minimal_comparison,
        )
        write_result(
            output / f"{benchmark}_compare_tok-native.json",
            native_comparison,
        )
        write_result(
            output / f"{benchmark}_compare_tok-tool-compatible.json",
            tool_compatible_comparison,
        )
        write_result(
            output / f"{benchmark}_compare_tok-neuro.json",
            neuro_comparison,
        )
        (output / f"{benchmark}_compare.md").write_text(
            render_comparison_markdown(
                baseline,
                [
                    minimal_comparison,
                    native_comparison,
                    tool_compatible_comparison,
                    neuro_comparison,
                ],
            )
        )
        if repeats > 1:
            stability_summary = summarize_compare_runs(repeated_results)
            (output / f"{benchmark}_stability.json").write_text(
                json.dumps(stability_summary, indent=2)
            )
            (output / f"{benchmark}_stability.md").write_text(
                render_stability_markdown(benchmark, model, stability_summary)
            )
        console.print(
            f"[green]✅ Live benchmark complete:[/green] {benchmark} "
            f"baseline_tokens={baseline.provider_usage.total_tokens} "
            f"tok_minimal_tokens={tok_minimal.provider_usage.total_tokens} "
            f"tok_native_tokens={tok_native.provider_usage.total_tokens} "
            f"tok_tool_compatible_tokens={tok_tool_compatible.provider_usage.total_tokens}"
        )
        console.print(
            f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_baseline.json'}"
        )
        console.print(
            f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_tok-minimal.json'}"
        )
        console.print(
            f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_tok-native.json'}"
        )
        console.print(
            "[cyan]Artifacts:[/cyan] "
            f"{output / f'{benchmark}_tok-tool-compatible.json'}"
        )
        console.print(
            f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_tok-neuro.json'}"
        )
        console.print(
            f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_compare.md'}"
        )
        if repeats > 1:
            console.print(
                f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_stability.json'}"
            )
            console.print(
                f"[cyan]Artifacts:[/cyan] {output / f'{benchmark}_stability.md'}"
            )
        console.print(f"[cyan]Best mode:[/cyan] {preferred_mode}")
        return

    if mode not in {
        "baseline",
        "tok-minimal",
        "tok-native",
        "tok-tool-compatible",
        "tok-neuro",
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
    console.print(
        f"[cyan]Artifact:[/cyan] {output / f'{benchmark}_{mode}.json'}"
    )


@dev_app.command("stress-language")
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
    """Run a long-lived Tok language stress harness against the OpenRouter path."""
    from ..stress_harness import (
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
            raise typer.BadParameter(
                f"--provider-options must be valid JSON: {exc}"
            ) from exc

    output_dir = output or default_output_dir()
    parsed_required_classes = (
        tuple(
            item.strip()
            for item in required_classes.split(",")
            if item.strip()
        )
        if required_classes
        else None
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
        required_classes=parsed_required_classes
        or StressHarnessConfig().required_classes,
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
        getattr(turn, "phase_name", "") == "retention-probe"
        and "retention_probe_early" in getattr(turn, "task_id", "")
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
    console.print(
        f"[cyan]Breakpoint classes:[/cyan] {', '.join(classes_seen) if classes_seen else 'none'}"
    )
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
        f"[cyan]Memory checks:[/cyan] reuse={result.reuse_checks_run} "
        f"checkpoint={result.checkpoint_checks_run}"
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
    console.print(
        f"[cyan]Early retention probe:[/cyan] {early_retention_probe_ran}"
    )
    console.print(
        f"[cyan]Late retention probe:[/cyan] {late_retention_probe_ran}"
    )
    console.print(
        f"[cyan]Resend modes:[/cyan] {', '.join(result.resend_modes_seen) or 'none'} "
        f"payload_pressure={result.payload_pressure_reached}"
    )
    console.print(
        f"[cyan]Compaction eligibility:[/cyan] {result.compaction_eligible}"
    )
    console.print(
        f"[cyan]Run diagnosis:[/cyan] {result.run_diagnosis} "
        f"weak_reasons={', '.join(result.weak_run_reasons) or 'none'}"
    )
    console.print(
        f"[cyan]First-anchor failure mode:[/cyan] {result.first_anchor_failure_mode}"
    )
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['stress_run']}")
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['breakpoints']}")
    console.print(f"[cyan]Artifacts:[/cyan] {artifacts['stress_report']}")
    console.print(
        f"[cyan]Artifacts:[/cyan] {artifacts['language_refactor_plan']}"
    )
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
    from ..analysis import run_dedup_frontier

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
