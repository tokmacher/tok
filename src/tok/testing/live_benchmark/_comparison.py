from __future__ import annotations

from typing import Any

from ._models import BenchmarkComparison, BenchmarkResult
from ._utils import _sum_warning_signals


def _diagnose_comparison(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    total_delta: int,
    reacquisition_delta: int,
    pressure_delta: int,
) -> str:
    if baseline.task_success and not candidate.task_success:
        return "lost_on_task_success"
    if total_delta < 0 and candidate.task_success:
        return "won_on_prompt_reduction"
    if reacquisition_delta > 0 and total_delta > 0:
        return "lost_on_reacquisition"

    response_signals = candidate.response_metrics.get("response_behavior_signals", {})
    if _sum_warning_signals(response_signals) > 0 and total_delta > 0:
        return "lost_on_response_drift"

    tok_overhead = int(candidate.prompt_metrics.get("tok_overhead_tokens", 0))
    total_saved = int(candidate.compression_metrics.get("total_saved_tokens", 0))
    if total_delta > 0 and tok_overhead >= total_saved:
        return "lost_on_bootstrap_overhead"

    if total_delta > 0 and pressure_delta > 0:
        return "lost_on_response_drift"

    return "mixed_result"


def compare_results(baseline: BenchmarkResult, candidate: BenchmarkResult) -> BenchmarkComparison:
    total_delta = candidate.provider_usage.total_tokens - baseline.provider_usage.total_tokens
    total_pct = None
    if baseline.provider_usage.total_tokens > 0:
        total_pct = round((total_delta / baseline.provider_usage.total_tokens) * 100.0, 1)

    baseline_reacq = int(baseline.response_metrics.get("reacquisition_cost_tokens", 0))
    candidate_reacq = int(candidate.response_metrics.get("reacquisition_cost_tokens", 0))
    baseline_pressure = int(baseline.response_metrics.get("invisible_pressure", 0))
    candidate_pressure = int(candidate.response_metrics.get("invisible_pressure", 0))
    task_success_equal_or_better = candidate.task_success and (
        baseline.task_success == candidate.task_success or not baseline.task_success
    )
    provider_total_token_winner = (
        candidate.mode if candidate.provider_usage.total_tokens < baseline.provider_usage.total_tokens else "baseline"
    )
    baseline_cost = baseline.provider_usage.cost_usd
    candidate_cost = candidate.provider_usage.cost_usd
    cost_delta_usd: float | None = None
    cost_delta_pct: float | None = None
    provider_cost_winner = "unknown"
    if baseline_cost is not None and candidate_cost is not None:
        cost_delta_usd = round(candidate_cost - baseline_cost, 6)
        if baseline_cost > 0:
            cost_delta_pct = round((cost_delta_usd / baseline_cost) * 100.0, 2)
        provider_cost_winner = candidate.mode if candidate_cost < baseline_cost else "baseline"

    fairness_diagnostics = _build_fairness_diagnostics(baseline, candidate)
    diagnosis = _diagnose_comparison(
        baseline,
        candidate,
        total_delta=total_delta,
        reacquisition_delta=candidate_reacq - baseline_reacq,
        pressure_delta=candidate_pressure - baseline_pressure,
    )
    tok_improved = task_success_equal_or_better and total_delta <= 0
    token_savings_without_cost_savings = bool(total_delta < 0 and cost_delta_usd is not None and cost_delta_usd >= 0)
    cost_savings_without_token_savings = bool(cost_delta_usd is not None and cost_delta_usd < 0 and total_delta >= 0)

    return BenchmarkComparison(
        benchmark=candidate.benchmark,
        model=candidate.model,
        candidate_mode=candidate.mode,
        baseline=baseline,
        candidate=candidate,
        prompt_token_delta=candidate.provider_usage.prompt_tokens - baseline.provider_usage.prompt_tokens,
        completion_token_delta=candidate.provider_usage.completion_tokens - baseline.provider_usage.completion_tokens,
        total_token_delta=total_delta,
        total_token_delta_pct=total_pct,
        latency_delta_ms=round(
            candidate.provider_usage.latency_ms - baseline.provider_usage.latency_ms,
            2,
        ),
        reacquisition_delta_tokens=candidate_reacq - baseline_reacq,
        pressure_delta=candidate_pressure - baseline_pressure,
        task_success_equal_or_better=task_success_equal_or_better,
        provider_total_token_winner=provider_total_token_winner,
        provider_cost_winner=provider_cost_winner,
        baseline_cost_usd=baseline_cost,
        candidate_cost_usd=candidate_cost,
        cost_delta_usd=cost_delta_usd,
        cost_delta_pct=cost_delta_pct,
        token_savings_without_cost_savings=token_savings_without_cost_savings,
        cost_savings_without_token_savings=cost_savings_without_token_savings,
        fairness_diagnostics=fairness_diagnostics,
        diagnosis=diagnosis,
        tok_improved=tok_improved,
    )


def select_preferred_mode(baseline: BenchmarkResult, comparisons: list[BenchmarkComparison]) -> str:
    viable = [comparison for comparison in comparisons if comparison.candidate.task_success]
    if not viable:
        return "baseline" if baseline.task_success else "none"
    best = min(
        viable,
        key=lambda comparison: comparison.candidate.provider_usage.total_tokens,
    )
    if not baseline.task_success:
        return best.candidate.mode
    if best.candidate.provider_usage.total_tokens < baseline.provider_usage.total_tokens:
        return best.candidate.mode
    return "baseline"


def _extract_result_warnings(result: BenchmarkResult) -> set[str]:
    warnings: set[str] = set()
    for turn in result.turns:
        diagnostics = turn.get("diagnostics", {}) if isinstance(turn, dict) else {}
        schema_forensics = diagnostics.get("schema_forensics", {}) if isinstance(diagnostics, dict) else {}
        for warning in schema_forensics.get("compatibility_warnings", []) or []:
            warnings.add(str(warning))
        for warning in diagnostics.get("tool_block_adaptation_warnings", []) or []:
            warnings.add(str(warning))
    return warnings


def _message_normalization_path(result: BenchmarkResult) -> str:
    raw = result.diagnostics.get("message_normalization_path", "")
    return str(raw).strip()


def _build_fairness_diagnostics(baseline: BenchmarkResult, candidate: BenchmarkResult) -> dict[str, Any]:
    baseline_path = _message_normalization_path(baseline)
    candidate_path = _message_normalization_path(candidate)
    baseline_warnings = sorted(_extract_result_warnings(baseline))
    candidate_warnings = sorted(_extract_result_warnings(candidate))

    asymmetry_flags: list[str] = []
    if baseline_path and candidate_path and baseline_path != candidate_path:
        asymmetry_flags.append("message_normalization_path_mismatch")
    if set(baseline_warnings) != set(candidate_warnings):
        asymmetry_flags.append("tool_block_warning_mismatch")

    return {
        "baseline_message_normalization_path": baseline_path,
        "candidate_message_normalization_path": candidate_path,
        "baseline_tool_block_warnings": baseline_warnings,
        "candidate_tool_block_warnings": candidate_warnings,
        "asymmetry_flags": asymmetry_flags,
    }
