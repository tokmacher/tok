from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ._comparison import compare_results, select_preferred_mode
from ._models import BenchmarkComparison, BenchmarkResult
from ._utils import _sum_warning_signals


def write_result(path: Path, payload: BenchmarkResult | BenchmarkComparison) -> None:
    path.write_text(json.dumps(payload.to_dict(), indent=2))


def render_comparison_markdown(baseline: BenchmarkResult, comparisons: list[BenchmarkComparison]) -> str:
    lines = [
        f"# Live Benchmark: {baseline.benchmark}",
        "",
        f"- Model: `{baseline.model}`",
        f"- Baseline total tokens: `{baseline.provider_usage.total_tokens}`",
        f"- Session turns: `{baseline.turn_count}`",
        "",
        "| Mode | Success | Total | Prompt | Completion | Tok saved | Tok overhead | Pressure | Reacquisition | Diagnosis |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| baseline | {baseline.task_success} | {baseline.provider_usage.total_tokens} | "
            f"{baseline.provider_usage.prompt_tokens} | {baseline.provider_usage.completion_tokens} | "
            f"0 | 0 | 0 | 0 | baseline |"
        ),
    ]
    for comparison in comparisons:
        candidate = comparison.candidate
        lines.append(
            f"| {candidate.mode} | {candidate.task_success} | {candidate.provider_usage.total_tokens} | "
            f"{candidate.provider_usage.prompt_tokens} | {candidate.provider_usage.completion_tokens} | "
            f"{candidate.compression_metrics.get('total_saved_tokens', 0)} | "
            f"{candidate.prompt_metrics.get('tok_overhead_tokens', 0)} | "
            f"{candidate.response_metrics.get('invisible_pressure', 0)} | "
            f"{candidate.response_metrics.get('reacquisition_cost_tokens', 0)} | "
            f"{comparison.diagnosis} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for comparison in comparisons:
        pct = f"{comparison.total_token_delta_pct:+.1f}%" if comparison.total_token_delta_pct is not None else "n/a"
        lines.extend(
            [
                f"### {comparison.candidate.mode}",
                "",
                f"- Total token delta: `{comparison.total_token_delta}` ({pct})",
                f"- Prompt token delta: `{comparison.prompt_token_delta}`",
                f"- Completion token delta: `{comparison.completion_token_delta}`",
                f"- Directive tokens estimate: `{comparison.candidate.prompt_metrics.get('directive_tokens_estimate', 0)}`",
                f"- State payload tokens estimate: `{comparison.candidate.prompt_metrics.get('state_payload_tokens_estimate', 0)}`",
                f"- Latency delta (ms): `{comparison.latency_delta_ms}`",
                f"- Reacquisition delta (tokens): `{comparison.reacquisition_delta_tokens}`",
                f"- Pressure delta: `{comparison.pressure_delta}`",
                f"- Task success equal or better: `{comparison.task_success_equal_or_better}`",
                f"- Candidate task success: `{comparison.candidate.task_success}`",
                f"- Candidate notes: `{', '.join(comparison.candidate.notes) or 'none'}`",
                f"- Provider total token winner: `{comparison.provider_total_token_winner}`",
                f"- Provider cost winner: `{comparison.provider_cost_winner}`",
                f"- Cost delta (USD): `{comparison.cost_delta_usd if comparison.cost_delta_usd is not None else 'n/a'}`",
                f"- Token savings without cost savings: `{comparison.token_savings_without_cost_savings}`",
                f"- Cost savings without token savings: `{comparison.cost_savings_without_token_savings}`",
                f"- Fairness asymmetry flags: `{', '.join(comparison.fairness_diagnostics.get('asymmetry_flags', [])) or 'none'}`",
                f"- Diagnosis: `{comparison.diagnosis}`",
                "",
            ]
        )
    lines.append(f"- Preferred mode: `{select_preferred_mode(baseline, comparisons)}`")
    lines.append("")
    return "\n".join(lines)


def summarize_compare_runs(
    repeated_results: list[dict[str, BenchmarkResult]],
    *,
    run_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not repeated_results:
        payload = {
            "runs": 0,
            "preferred_mode_counts": {},
            "mode_summaries": {},
        }
        if run_fingerprint is not None:
            payload["run_fingerprint"] = run_fingerprint
        return payload

    mode_order = ("baseline", "tok-universal")
    preferred_mode_counts: dict[str, int] = {}
    mode_summaries: dict[str, Any] = {}
    comparisons_all: list[BenchmarkComparison] = []

    for run in repeated_results:
        baseline = run["baseline"]
        candidate = run.get("tok-universal") or run.get("tok-tool-compatible")
        if candidate is None:
            continue
        comparisons = [compare_results(baseline, candidate)]
        comparisons_all.extend(comparisons)
        if not comparisons:
            continue
        preferred = select_preferred_mode(baseline, comparisons)
        preferred_mode_counts[preferred] = preferred_mode_counts.get(preferred, 0) + 1

    for mode in mode_order:
        if mode == "tok-universal":
            results = [
                (run.get("tok-universal") or run.get("tok-tool-compatible"))
                for run in repeated_results
                if (run.get("tok-universal") or run.get("tok-tool-compatible")) is not None
            ]
        else:
            results = [run[mode] for run in repeated_results if mode in run]
        if not results:
            continue
        total_tokens = [result.provider_usage.total_tokens for result in results if result is not None]
        prompt_tokens = [result.provider_usage.prompt_tokens for result in results if result is not None]
        completion_tokens = [result.provider_usage.completion_tokens for result in results if result is not None]
        latency_ms = [result.provider_usage.latency_ms for result in results if result is not None]
        successes = sum(1 for result in results if result is not None and result.task_success)

        mode_summaries[mode] = {
            "runs": len(results),
            "success_rate": round(successes / len(results), 3),
            "success_count": successes,
            "median_total_tokens": int(statistics.median(total_tokens)),
            "min_total_tokens": min(total_tokens),
            "max_total_tokens": max(total_tokens),
            "median_prompt_tokens": int(statistics.median(prompt_tokens)),
            "median_completion_tokens": int(statistics.median(completion_tokens)),
            "median_latency_ms": round(statistics.median(latency_ms), 2),
            "total_token_variance": round(float(statistics.pvariance(total_tokens)), 2)
            if len(total_tokens) > 1
            else 0.0,
        }

    total_comparisons = len(comparisons_all)
    token_win_count = sum(
        1
        for comparison in comparisons_all
        if comparison.total_token_delta < 0 and comparison.task_success_equal_or_better
    )
    cost_comparable = [comparison for comparison in comparisons_all if comparison.cost_delta_usd is not None]
    cost_win_count = sum(
        1 for comparison in cost_comparable if comparison.cost_delta_usd is not None and comparison.cost_delta_usd < 0
    )
    dominant_mode = ""
    dominant_rate = 0.0
    if preferred_mode_counts:
        dominant_mode, dominant_count = max(preferred_mode_counts.items(), key=lambda item: item[1])
        dominant_rate = round(dominant_count / max(1, len(repeated_results)), 3)

    payload = {
        "runs": len(repeated_results),
        "preferred_mode_counts": preferred_mode_counts,
        "mode_summaries": mode_summaries,
        "confidence_summary": {
            "dominant_preferred_mode": dominant_mode,
            "dominant_preferred_rate": dominant_rate,
            "token_win_rate": round(token_win_count / total_comparisons, 3) if total_comparisons else 0.0,
            "cost_win_rate": round(cost_win_count / len(cost_comparable), 3) if cost_comparable else None,
        },
    }
    if run_fingerprint is not None:
        payload["run_fingerprint"] = run_fingerprint
    return payload


def summarize_compare_triage(
    repeated_results: list[dict[str, BenchmarkResult]],
    *,
    run_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode_order = ("baseline", "tok-universal")
    mode_summaries: dict[str, Any] = {}
    comparisons_all: list[BenchmarkComparison] = []
    fairness_flag_counter: Counter[str] = Counter()

    for mode in mode_order:
        if mode == "tok-universal":
            results = [
                run.get("tok-universal") or run.get("tok-tool-compatible")
                for run in repeated_results
                if (run.get("tok-universal") or run.get("tok-tool-compatible")) is not None
            ]
        else:
            results = [run.get("baseline") for run in repeated_results if run.get("baseline") is not None]
        if not results:
            continue

        legacy_success_count = sum(1 for result in results if result and result.task_success)
        repo_success_count = sum(
            1
            for result in results
            if result and bool(result.diagnostics.get("repo_grounded_task_success", result.task_success))
        )
        failure_counter: Counter[str] = Counter()
        response_contract_friction_runs = 0
        response_contract_friction_signals = 0
        for result in results:
            if result is None:
                continue
            for reason in result.notes:
                if isinstance(reason, str) and reason.startswith("repo_grounded:"):
                    continue
                failure_counter[str(reason)] += 1
            for reason in result.diagnostics.get("repo_grounded_failures", []):
                failure_counter[f"repo_grounded:{reason}"] += 1
            warnings = _sum_warning_signals(result.response_metrics.get("response_behavior_signals", {}))
            response_contract_friction_signals += warnings
            if warnings > 0:
                response_contract_friction_runs += 1

        top_failures = [{"reason": reason, "count": count} for reason, count in failure_counter.most_common(10)]
        mode_summaries[mode] = {
            "runs": len(results),
            "legacy_success_rate": round(legacy_success_count / len(results), 3),
            "legacy_success_count": legacy_success_count,
            "repo_grounded_success_rate": round(repo_success_count / len(results), 3),
            "repo_grounded_success_count": repo_success_count,
            "response_contract_friction_runs": response_contract_friction_runs,
            "response_contract_friction_signals": response_contract_friction_signals,
            "top_failure_reasons": top_failures,
        }

    for run in repeated_results:
        baseline = run.get("baseline")
        candidate = run.get("tok-universal") or run.get("tok-tool-compatible")
        if baseline is None or candidate is None:
            continue
        comparison = compare_results(baseline, candidate)
        comparisons_all.append(comparison)
        for flag in comparison.fairness_diagnostics.get("asymmetry_flags", []):
            fairness_flag_counter[str(flag)] += 1

    total_comparisons = len(comparisons_all)
    token_win_count = sum(1 for comparison in comparisons_all if comparison.total_token_delta < 0)
    token_without_cost_count = sum(1 for comparison in comparisons_all if comparison.token_savings_without_cost_savings)
    cost_comparable = [comparison for comparison in comparisons_all if comparison.cost_delta_usd is not None]
    cost_win_count = sum(
        1 for comparison in cost_comparable if comparison.cost_delta_usd is not None and comparison.cost_delta_usd < 0
    )
    dominant_mode = ""
    dominant_rate = 0.0
    preferred_mode_counts: Counter[str] = Counter()
    for run in repeated_results:
        baseline = run.get("baseline")
        candidate = run.get("tok-universal") or run.get("tok-tool-compatible")
        if baseline is None or candidate is None:
            continue
        preferred_mode_counts[select_preferred_mode(baseline, [compare_results(baseline, candidate)])] += 1
    if preferred_mode_counts:
        dominant_mode, dominant_count = preferred_mode_counts.most_common(1)[0]
        dominant_rate = round(dominant_count / max(1, len(repeated_results)), 3)

    payload = {
        "runs": len(repeated_results),
        "mode_summaries": mode_summaries,
        "fairness_summary": {
            "asymmetry_runs": sum(
                1 for comparison in comparisons_all if comparison.fairness_diagnostics.get("asymmetry_flags")
            ),
            "asymmetry_flag_counts": dict(fairness_flag_counter),
        },
        "cost_summary": {
            "token_win_rate": round(token_win_count / total_comparisons, 3) if total_comparisons else 0.0,
            "cost_win_rate": round(cost_win_count / len(cost_comparable), 3) if cost_comparable else None,
            "token_win_without_cost_win_runs": token_without_cost_count,
        },
        "repeat_confidence": {
            "dominant_preferred_mode": dominant_mode,
            "dominant_preferred_rate": dominant_rate,
        },
    }
    if run_fingerprint is not None:
        payload["run_fingerprint"] = run_fingerprint
    return payload
