"""Backward-compatible shim for live benchmark helpers."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

# Re-export all canonical symbols for backward compatibility
from .testing.live_benchmark import (  # noqa: F401
    BenchmarkResult,
    ProviderUsageSnapshot,
    _chunk_messages,
    _turn_prompts,
    normalize_fixture_messages,
)


def _impl() -> Any:
    return import_module(".testing.live_benchmark", __package__)


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    fixture_path: Path
    system_prompt: str
    followup_prompt: str
    success_terms: tuple[str, ...]
    min_success_terms: int = 2
    expected_file_terms: tuple[str, ...] = ()
    expected_verification_terms: tuple[str, ...] = ()
    default_turns: int = 3
    prompt_sequence: tuple[str, ...] = ()


def load_benchmark_definition(name: str) -> BenchmarkDefinition:
    return cast(BenchmarkDefinition, _impl().load_benchmark_definition(name))


class LiveBenchmarkRunner:
    def __new__(cls, *args: Any, **kwargs: Any) -> LiveBenchmarkRunner:
        return _impl().LiveBenchmarkRunner(*args, **kwargs)  # type: ignore[no-any-return]

    def run(
        self, definition: BenchmarkDefinition, *, mode: str, turns: int = 3
    ) -> BenchmarkResult:
        raise NotImplementedError


def compare_results(baseline: Any, candidate: Any) -> Any:
    return _impl().compare_results(baseline, candidate)


def select_preferred_mode(baseline: Any, comparisons: list[Any]) -> str:
    viable = [
        comparison
        for comparison in comparisons
        if comparison.candidate.task_success
    ]
    if not viable:
        return "baseline" if baseline.task_success else "none"
    best = min(
        viable,
        key=lambda comparison: (
            comparison.candidate.provider_usage.total_tokens
        ),
    )
    if not baseline.task_success:
        return cast(str, best.candidate.mode)
    if (
        best.candidate.provider_usage.total_tokens
        < baseline.provider_usage.total_tokens
    ):
        return cast(str, best.candidate.mode)
    return "baseline"


def write_result(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload.to_dict(), indent=2))


def render_comparison_markdown(baseline: Any, comparisons: list[Any]) -> str:
    return cast(str, _impl().render_comparison_markdown(baseline, comparisons))


def summarize_compare_runs(
    repeated_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not repeated_results:
        return {"runs": 0, "preferred_mode_counts": {}, "mode_summaries": {}}

    mode_order = (
        "baseline",
        "tok-minimal",
        "tok-native",
        "tok-tool-compatible",
        "tok-neuro",
    )
    preferred_mode_counts: dict[str, int] = {}
    mode_summaries: dict[str, Any] = {}

    for run in repeated_results:
        baseline = run["baseline"]
        comparisons = []
        for mode in (
            "tok-minimal",
            "tok-native",
            "tok-tool-compatible",
            "tok-neuro",
        ):
            if mode in run:
                comparisons.append(compare_results(baseline, run[mode]))
        if comparisons:
            preferred = select_preferred_mode(baseline, comparisons)
            preferred_mode_counts[preferred] = (
                preferred_mode_counts.get(preferred, 0) + 1
            )

    for mode in mode_order:
        results = [run[mode] for run in repeated_results if mode in run]
        if not results:
            continue
        total_tokens = [
            result.provider_usage.total_tokens for result in results
        ]
        prompt_tokens = [
            result.provider_usage.prompt_tokens for result in results
        ]
        completion_tokens = [
            result.provider_usage.completion_tokens for result in results
        ]
        latency_ms = [result.provider_usage.latency_ms for result in results]
        successes = sum(1 for result in results if result.task_success)
        mode_summaries[mode] = {
            "runs": len(results),
            "success_rate": round(successes / len(results), 3),
            "success_count": successes,
            "median_total_tokens": int(statistics.median(total_tokens)),
            "min_total_tokens": min(total_tokens),
            "max_total_tokens": max(total_tokens),
            "median_prompt_tokens": int(statistics.median(prompt_tokens)),
            "median_completion_tokens": int(
                statistics.median(completion_tokens)
            ),
            "median_latency_ms": round(statistics.median(latency_ms), 2),
        }

    return {
        "runs": len(repeated_results),
        "preferred_mode_counts": preferred_mode_counts,
        "mode_summaries": mode_summaries,
    }


def render_stability_markdown(
    benchmark: str,
    model: str,
    summary: dict[str, Any],
) -> str:
    lines = [
        f"# Live Benchmark Stability: {benchmark}",
        "",
        f"- Model: `{model}`",
        f"- Repeats: `{summary.get('runs', 0)}`",
        "",
        "## Preferred Mode Counts",
        "",
    ]
    preferred_counts = summary.get("preferred_mode_counts", {})
    if preferred_counts:
        for mode, count in sorted(preferred_counts.items()):
            lines.append(f"- `{mode}`: `{count}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Median Metrics",
            "",
            "| Mode | Success Rate | Median Total | Min | Max | Median Prompt | Median Completion | Median Latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, metrics in summary.get("mode_summaries", {}).items():
        lines.append(
            f"| {mode} | {metrics['success_rate']:.3f} | {metrics['median_total_tokens']} | "
            f"{metrics['min_total_tokens']} | {metrics['max_total_tokens']} | "
            f"{metrics['median_prompt_tokens']} | {metrics['median_completion_tokens']} | "
            f"{metrics['median_latency_ms']} |"
        )

    lines.append("")
    return "\n".join(lines)
