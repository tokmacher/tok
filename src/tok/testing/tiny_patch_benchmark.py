"""Tiered patch benchmark workflow focused on completion parity and token savings."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tok.testing.benchmark_executor import CatalogBenchmarkRun, run_catalog_benchmark_suite
from tok.testing.benchmark_suite import BenchmarkCatalog, BenchmarkLane, BenchmarkTaskManifest
from tok.testing.live_benchmark import LiveBenchmarkRunner

PATCH_SUITE_BENCHMARKS = {"tiny-patch", "small-patch", "medium-patch", "large-patch"}
_MIN_SUCCESS_FLOOR = 0.5


def is_patch_suite_benchmark(name: str) -> bool:
    return name in PATCH_SUITE_BENCHMARKS


def _production_lane() -> BenchmarkLane:
    return BenchmarkLane.from_dict(
        {
            "id": "production_claude_lane",
            "runtime_path": (
                "Bridge-first Claude Code flow through UniversalTokRuntime using the production default request/response path."
            ),
            "transport_shape": "claude_code_bridge_messages",
            "model_family": "claude",
            "provider": "anthropic",
            "adapter_name": "",
            "adapter_notes": "",
            "claim_scope": "headline",
            "normalized_differences": [],
        }
    )


def _task(
    *,
    task_id: str,
    title: str,
    summary: str,
    prompt: str,
    workspace_path: str,
    hidden_test: str,
    allowed_paths: tuple[str, ...],
    step_budget: int,
    time_budget_minutes: int,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "family": "execution_patch",
        "title": title,
        "summary": summary,
        "repo": "local/patch_suite",
        "ref": "HEAD",
        "setup_script": "no_setup_required",
        "prompt": prompt,
        "allowed_tools": ["list_dir", "view_file", "grep_search", "edit_file", "run_tests"],
        "time_budget_minutes": time_budget_minutes,
        "step_budget": step_budget,
        "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
        "artifact_policy": {"publish_diff": True},
        "public_release": False,
        "asset_dir": "tests/fixtures/patch_suites",
        "workspace_source": {"kind": "local_checkout", "path": workspace_path},
        "family_payload": {
            "allowed_paths": list(allowed_paths),
            "visible_tests": [],
            "hidden_tests": [hidden_test],
        },
        "allowed_paths": list(allowed_paths),
        "visible_tests": [],
        "hidden_tests": [hidden_test],
    }


def _suite_payloads(benchmark_name: str) -> tuple[dict[str, Any], ...]:
    if benchmark_name == "tiny-patch":
        return (
            _task(
                task_id="tiny.calc.add",
                title="Fix add()",
                summary="Repair add so it returns a plus b.",
                prompt="Fix add(a, b) in src/calculator.py so tests pass. Only edit the allowed file.",
                workspace_path="tests/fixtures/tiny_patch/workspaces/calc_add",
                hidden_test="tests/test_calculator.py::test_add_two_positive_numbers",
                allowed_paths=("src/calculator.py",),
                step_budget=25,
                time_budget_minutes=1,
            ),
            _task(
                task_id="tiny.calc.subtract",
                title="Fix subtract()",
                summary="Repair subtract so it returns a minus b.",
                prompt="Fix subtract(a, b) in src/calculator.py so tests pass. Only edit the allowed file.",
                workspace_path="tests/fixtures/tiny_patch/workspaces/calc_subtract",
                hidden_test="tests/test_calculator.py::test_subtract_two_numbers",
                allowed_paths=("src/calculator.py",),
                step_budget=25,
                time_budget_minutes=1,
            ),
            _task(
                task_id="tiny.calc.multiply",
                title="Fix multiply()",
                summary="Repair multiply so it returns a times b.",
                prompt="Fix multiply(a, b) in src/calculator.py so tests pass. Only edit the allowed file.",
                workspace_path="tests/fixtures/tiny_patch/workspaces/calc_multiply",
                hidden_test="tests/test_calculator.py::test_multiply_two_numbers",
                allowed_paths=("src/calculator.py",),
                step_budget=25,
                time_budget_minutes=1,
            ),
            _task(
                task_id="tiny.calc.divide-zero",
                title="Fix divide() zero handling",
                summary="Raise ValueError on division by zero.",
                prompt="Fix divide(a, b) so division by zero raises ValueError. Only edit the allowed file.",
                workspace_path="tests/fixtures/tiny_patch/workspaces/calc_divide_zero",
                hidden_test="tests/test_calculator.py::test_divide_by_zero_raises",
                allowed_paths=("src/calculator.py",),
                step_budget=25,
                time_budget_minutes=1,
            ),
            _task(
                task_id="tiny.calc.percent",
                title="Fix percent_of()",
                summary="Return fractional percentage correctly.",
                prompt="Fix percent_of(value, percent) so tests pass. Only edit the allowed file.",
                workspace_path="tests/fixtures/tiny_patch/workspaces/calc_percent",
                hidden_test="tests/test_calculator.py::test_percent_supports_fractional_result",
                allowed_paths=("src/calculator.py",),
                step_budget=25,
                time_budget_minutes=1,
            ),
        )

    if benchmark_name == "small-patch":
        return (
            _task(
                task_id="small.invoice.totals",
                title="Fix invoice totals",
                summary="Repair invoice adjustments and total ordering across pricing modules.",
                prompt=(
                    "Fix invoice pricing behavior across subtotal, discount, tax, and service-fee paths "
                    "so hidden tests pass."
                ),
                workspace_path="tests/fixtures/patch_suites/small/workspaces/invoice_totals",
                hidden_test="tests/test_pricing.py",
                allowed_paths=("src/pricing.py", "src/adjustments.py"),
                step_budget=30,
                time_budget_minutes=3,
            ),
            _task(
                task_id="small.text.pipeline",
                title="Fix text pipeline",
                summary="Repair normalization, tokenization, filtering, and dedupe ordering for user text.",
                prompt="Fix text pipeline behavior across normalize/tokenize/filter/prepare modules so tests pass.",
                workspace_path="tests/fixtures/patch_suites/small/workspaces/text_pipeline",
                hidden_test="tests/test_text_pipeline.py",
                allowed_paths=("src/normalize.py", "src/tokenize.py", "src/pipeline.py"),
                step_budget=30,
                time_budget_minutes=3,
            ),
            _task(
                task_id="small.auth.rules",
                title="Fix auth rules",
                summary="Repair access control, reset policy, scopes, and session issuance guardrails.",
                prompt="Fix auth policy behavior across access, reset, scopes, and session token checks.",
                workspace_path="tests/fixtures/patch_suites/small/workspaces/auth_rules",
                hidden_test="tests/test_auth.py",
                allowed_paths=("src/auth.py", "src/policy.py", "src/session.py"),
                step_budget=30,
                time_budget_minutes=3,
            ),
        )

    if benchmark_name == "medium-patch":
        return (
            _task(
                task_id="medium.ledger.reconcile",
                title="Fix ledger reconciliation",
                summary="Repair normalization, deduplication, and report rounding.",
                prompt="Fix reconciliation/report behavior so tests pass with stable totals.",
                workspace_path="tests/fixtures/patch_suites/medium/workspaces/ledger_reconcile",
                hidden_test="tests/test_reconcile.py",
                allowed_paths=("src/ledger.py", "src/reconcile.py", "src/report.py"),
                step_budget=35,
                time_budget_minutes=4,
            ),
            _task(
                task_id="medium.api.sanitizer",
                title="Fix API sanitizer",
                summary="Repair email/username sanitation and validation thresholds.",
                prompt="Fix sanitizer and validation behavior so handler tests pass.",
                workspace_path="tests/fixtures/patch_suites/medium/workspaces/api_sanitizer",
                hidden_test="tests/test_handlers.py",
                allowed_paths=("src/sanitize.py", "src/validators.py", "src/handlers.py"),
                step_budget=35,
                time_budget_minutes=4,
            ),
            _task(
                task_id="medium.rule.engine",
                title="Fix rule engine",
                summary="Repair parse/load/evaluation semantics for ordered threshold rules.",
                prompt="Fix rule parsing/loading/evaluation so rules tests pass.",
                workspace_path="tests/fixtures/patch_suites/medium/workspaces/rule_engine",
                hidden_test="tests/test_rules.py",
                allowed_paths=("src/rules.py", "src/loader.py", "src/evaluator.py"),
                step_budget=35,
                time_budget_minutes=4,
            ),
        )

    if benchmark_name == "large-patch":
        return (
            _task(
                task_id="large.inventory.allocator",
                title="Fix inventory allocator",
                summary="Repair planning math across forecast, allocation, and reorder logic.",
                prompt="Fix inventory planning behavior so tests pass with correct required, allocation, and reorder values.",
                workspace_path="tests/fixtures/patch_suites/large/workspaces/inventory_allocator",
                hidden_test="tests/test_planner.py",
                allowed_paths=("src/forecast.py", "src/allocator.py", "src/planner.py"),
                step_budget=40,
                time_budget_minutes=5,
            ),
            _task(
                task_id="large.billing.rollup",
                title="Fix billing rollup",
                summary="Repair normalization, rate lookup, rollup math, and invoice rendering.",
                prompt="Fix billing rollup logic so invoices match expected totals and formatting.",
                workspace_path="tests/fixtures/patch_suites/large/workspaces/billing_rollup",
                hidden_test="tests/test_invoice.py",
                allowed_paths=("src/usage.py", "src/rollup.py", "src/invoice.py"),
                step_budget=40,
                time_budget_minutes=5,
            ),
            _task(
                task_id="large.workflow.router",
                title="Fix workflow router",
                summary="Repair parser normalization, queue routing rules, and SLA policy.",
                prompt="Fix routing behavior for priority/team workflows so tests pass.",
                workspace_path="tests/fixtures/patch_suites/large/workspaces/workflow_router",
                hidden_test="tests/test_router.py",
                allowed_paths=("src/parser.py", "src/rules.py", "src/router.py"),
                step_budget=40,
                time_budget_minutes=5,
            ),
        )

    msg = f"Unknown patch benchmark suite: {benchmark_name}"
    raise ValueError(msg)


def load_patch_suite_catalog(*, benchmark_name: str, repo_root: Path) -> BenchmarkCatalog:
    tasks = tuple(BenchmarkTaskManifest.from_dict(payload) for payload in _suite_payloads(benchmark_name))
    return BenchmarkCatalog(root=str(repo_root.resolve()), lanes=(_production_lane(),), tasks=tasks)


def load_tiny_patch_catalog(*, repo_root: Path) -> BenchmarkCatalog:
    return load_patch_suite_catalog(benchmark_name="tiny-patch", repo_root=repo_root)


def _collect_completion_failure_reasons(tasks_root: Path) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    if not tasks_root.exists():
        return []
    for run_path in tasks_root.glob("*/repeat_*/baseline/run.json"):
        payload = json.loads(run_path.read_text())
        if bool(payload.get("evaluation", {}).get("success")):
            continue
        for note in payload.get("evaluation", {}).get("notes", []):
            counts[f"baseline:{note}"] += 1
    for run_path in tasks_root.glob("*/repeat_*/tok-universal/run.json"):
        payload = json.loads(run_path.read_text())
        if bool(payload.get("evaluation", {}).get("success")):
            continue
        for note in payload.get("evaluation", {}).get("notes", []):
            counts[f"tok-universal:{note}"] += 1
    return [{"reason": reason, "count": count} for reason, count in counts.most_common(10)]


def _load_condition_runs(tasks_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for condition in ("baseline", "tok-universal"):
        for run_path in tasks_root.glob(f"*/repeat_*/{condition}/run.json"):
            payload = json.loads(run_path.read_text())
            task_id = run_path.parents[2].name
            repeat_raw = run_path.parents[1].name
            repeat_index = 1
            if repeat_raw.startswith("repeat_"):
                try:
                    repeat_index = int(repeat_raw.split("_", 1)[1])
                except ValueError:
                    repeat_index = 1
            runs.append(
                {
                    "task_id": task_id,
                    "repeat_index": repeat_index,
                    "condition": condition,
                    "payload": payload,
                }
            )
    return runs


def _turn_diagnostics(
    *,
    task_runs_root: Path,
    comparison_runs: list[Any],
) -> dict[str, Any]:
    loaded_runs = _load_condition_runs(task_runs_root)
    if not loaded_runs:
        return {}

    by_condition: dict[str, list[dict[str, Any]]] = {"baseline": [], "tok-universal": []}
    step_totals: dict[str, dict[int, list[int]]] = {"baseline": defaultdict(list), "tok-universal": defaultdict(list)}
    tok_signal_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    totals_by_key: dict[tuple[str, int, str], int] = {}

    for item in loaded_runs:
        payload = item["payload"]
        turns = payload.get("turns") or []
        provider_usage = payload.get("provider_usage") or {}
        total_tokens = int(provider_usage.get("total_tokens", 0) or 0)
        prompt_tokens = int(provider_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(provider_usage.get("completion_tokens", 0) or 0)
        turn_count = len(turns)
        condition = item["condition"]
        by_condition[condition].append(
            {
                "task_id": item["task_id"],
                "repeat_index": item["repeat_index"],
                "turn_count": turn_count,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tool_calls": int(payload.get("tool_calls", 0) or 0),
            }
        )
        totals_by_key[(item["task_id"], item["repeat_index"], condition)] = total_tokens

        for index, turn in enumerate(turns, start=1):
            turn_total = int((turn.get("provider_usage") or {}).get("total_tokens", 0) or 0)
            step_totals[condition][index].append(turn_total)
            if condition != "tok-universal":
                continue
            signals = (turn.get("response_metrics") or {}).get("response_behavior_signals") or {}
            for key, value in signals.items():
                if isinstance(value, (int, float)) and value:
                    tok_signal_counts[key] += int(value)
            if int(signals.get("tok_bridge_strict_failure", 0) or 0) > 0:
                warning_counts["tok_bridge_strict_failure"] += 1
            if int(signals.get("tok_bridge_strict_missing_max_tokens", 0) or 0) > 0:
                warning_counts["tok_bridge_strict_missing_max_tokens"] += 1

        for note in payload.get("notes") or []:
            if isinstance(note, str) and note.startswith("text_tool_extraction_step_"):
                warning_counts["text_tool_extraction_used"] += 1

    def _averages(items: list[dict[str, Any]]) -> dict[str, float]:
        if not items:
            return {
                "avg_turns": 0.0,
                "avg_tokens_per_turn": 0.0,
                "avg_prompt_tokens_per_turn": 0.0,
                "avg_completion_tokens_per_turn": 0.0,
                "avg_tool_calls": 0.0,
            }
        turn_sum = sum(item["turn_count"] for item in items)
        total_sum = sum(item["total_tokens"] for item in items)
        prompt_sum = sum(item["prompt_tokens"] for item in items)
        completion_sum = sum(item["completion_tokens"] for item in items)
        tool_sum = sum(item["tool_calls"] for item in items)
        turn_denominator = max(1, turn_sum)
        return {
            "avg_turns": round(turn_sum / len(items), 2),
            "avg_tokens_per_turn": round(total_sum / turn_denominator, 1),
            "avg_prompt_tokens_per_turn": round(prompt_sum / turn_denominator, 1),
            "avg_completion_tokens_per_turn": round(completion_sum / turn_denominator, 1),
            "avg_tool_calls": round(tool_sum / len(items), 2),
        }

    per_step_medians: dict[str, dict[str, float]] = {"baseline": {}, "tok-universal": {}}
    for condition in ("baseline", "tok-universal"):
        for step, values in step_totals[condition].items():
            if not values:
                continue
            per_step_medians[condition][str(step)] = round(float(statistics.median(values)), 1)

    step_delta_tokens: dict[str, float] = {}
    step_delta_pct: dict[str, float] = {}
    cumulative_baseline = 0.0
    cumulative_tok = 0.0
    cumulative_step_delta_tokens: dict[str, float] = {}
    first_cumulative_parity_turn: int | None = None
    all_steps = sorted({*step_totals["baseline"].keys(), *step_totals["tok-universal"].keys()})
    for step in all_steps:
        baseline_median = per_step_medians["baseline"].get(str(step))
        tok_median = per_step_medians["tok-universal"].get(str(step))
        if baseline_median is None or tok_median is None:
            continue
        delta_tokens = round(float(tok_median - baseline_median), 1)
        step_delta_tokens[str(step)] = delta_tokens
        if baseline_median > 0:
            step_delta_pct[str(step)] = round((delta_tokens / baseline_median) * 100.0, 1)
        cumulative_baseline += baseline_median
        cumulative_tok += tok_median
        cumulative_delta = round(float(cumulative_tok - cumulative_baseline), 1)
        cumulative_step_delta_tokens[str(step)] = cumulative_delta
        if first_cumulative_parity_turn is None and cumulative_tok <= cumulative_baseline:
            first_cumulative_parity_turn = step

    top_delta_tasks: list[dict[str, Any]] = []
    for run in comparison_runs:
        key = (run.task_id, run.repeat_index)
        baseline_total = totals_by_key.get((key[0], key[1], "baseline"))
        tok_total = totals_by_key.get((key[0], key[1], "tok-universal"))
        if baseline_total is None or tok_total is None:
            continue
        baseline_turns = next(
            (
                item["turn_count"]
                for item in by_condition["baseline"]
                if item["task_id"] == key[0] and item["repeat_index"] == key[1]
            ),
            0,
        )
        tok_turns = next(
            (
                item["turn_count"]
                for item in by_condition["tok-universal"]
                if item["task_id"] == key[0] and item["repeat_index"] == key[1]
            ),
            0,
        )
        top_delta_tasks.append(
            {
                "task_id": run.task_id,
                "repeat_index": run.repeat_index,
                "delta_tokens": tok_total - baseline_total,
                "delta_turns": tok_turns - baseline_turns,
            }
        )
    top_delta_tasks.sort(key=lambda item: abs(int(item["delta_tokens"])), reverse=True)

    return {
        "baseline": _averages(by_condition["baseline"]),
        "tok-universal": _averages(by_condition["tok-universal"]),
        "median_step_total_tokens": per_step_medians,
        "median_step_token_delta": step_delta_tokens,
        "median_step_token_delta_pct": step_delta_pct,
        "cumulative_median_step_token_delta": cumulative_step_delta_tokens,
        "first_cumulative_parity_turn": first_cumulative_parity_turn,
        "top_tok_response_signals": [
            {"signal": signal, "count": count} for signal, count in tok_signal_counts.most_common(12)
        ],
        "runtime_contract_warnings": dict(warning_counts),
        "top_token_delta_tasks": top_delta_tasks[:10],
    }


def summarize_patch_suite_run(
    catalog_run: CatalogBenchmarkRun,
    *,
    output_root: Path,
    benchmark_name: str,
) -> dict[str, Any]:
    runs = list(catalog_run.runs)
    if not runs:
        return {
            "benchmark": benchmark_name,
            "runs": 0,
            "completion_success_rate": {"baseline": 0.0, "tok-universal": 0.0},
            "matched_completion_pair_count": 0,
            "matched_pair_token_delta_median": None,
            "claimable": False,
            "top_completion_failure_reasons": [],
            "advisory_warning_counts": {},
            "tool_engagement_stats": {},
        }

    baseline_rate = round(sum(1 for run in runs if run.baseline_success) / len(runs), 3)
    tok_rate = round(sum(1 for run in runs if run.tok_success) / len(runs), 3)
    matched_deltas = [run.total_token_delta for run in runs if run.matched_completion_pair]
    task_runs_root = output_root / "tasks"
    turn_diagnostics = _turn_diagnostics(task_runs_root=task_runs_root, comparison_runs=runs)
    baseline_totals = {
        (item["task_id"], item["repeat_index"], item["condition"]): int(
            (item["payload"].get("provider_usage") or {}).get("total_tokens", 0) or 0
        )
        for item in _load_condition_runs(task_runs_root)
    }
    matched_delta_pct = []
    for run in runs:
        if not run.matched_completion_pair:
            continue
        baseline_total = baseline_totals.get((run.task_id, run.repeat_index, "baseline"), 0)
        if baseline_total <= 0:
            continue
        matched_delta_pct.append((float(run.total_token_delta) / float(baseline_total)) * 100.0)
    advisory_counts: Counter[str] = Counter()
    for run in runs:
        for violation in run.format_contract_violations:
            advisory_counts[violation] += 1
        if run.invalid_tool_calls > 0:
            advisory_counts["invalid_tool_calls_present"] += 1
        if run.baseline_tool_calls == 0:
            advisory_counts["baseline_zero_tool_calls"] += 1
        if run.tok_tool_calls == 0:
            advisory_counts["tok_zero_tool_calls"] += 1

    both_fail_count = sum(1 for run in runs if not run.baseline_success and not run.tok_success)
    claimable = (
        tok_rate >= _MIN_SUCCESS_FLOOR
        and tok_rate >= baseline_rate
        and both_fail_count < len(runs)
        and len(matched_deltas) > 0
    )

    baseline_tool_calls = [run.baseline_tool_calls for run in runs]
    tok_tool_calls = [run.tok_tool_calls for run in runs]

    return {
        "benchmark": benchmark_name,
        "runs": len(runs),
        "completion_success_rate": {"baseline": baseline_rate, "tok-universal": tok_rate},
        "matched_completion_pair_count": len(matched_deltas),
        "matched_pair_token_delta_median": (
            round(float(statistics.median(matched_deltas)), 1) if matched_deltas else None
        ),
        "matched_pair_token_delta_pct_median": (
            round(float(statistics.median(matched_delta_pct)), 1) if matched_delta_pct else None
        ),
        "claimable": claimable,
        "top_completion_failure_reasons": _collect_completion_failure_reasons(task_runs_root),
        "advisory_warning_counts": dict(advisory_counts),
        "tool_engagement_stats": {
            "baseline_avg_tool_calls": round(sum(baseline_tool_calls) / len(baseline_tool_calls), 2),
            "tok_avg_tool_calls": round(sum(tok_tool_calls) / len(tok_tool_calls), 2),
        },
        "turn_diagnostics": turn_diagnostics,
    }


def summarize_tiny_patch_run(catalog_run: CatalogBenchmarkRun, *, output_root: Path) -> dict[str, Any]:
    return summarize_patch_suite_run(catalog_run, output_root=output_root, benchmark_name="tiny-patch")


def render_patch_suite_markdown(summary: dict[str, Any], *, benchmark_name: str) -> str:
    completion = summary.get("completion_success_rate", {})
    lines = [
        f"# Patch Benchmark Summary: {benchmark_name}",
        "",
        f"- Runs: `{summary.get('runs', 0)}`",
        f"- Baseline completion success rate: `{completion.get('baseline', 0.0):.3f}`",
        f"- Tok completion success rate: `{completion.get('tok-universal', 0.0):.3f}`",
        f"- Matched completion pairs: `{summary.get('matched_completion_pair_count', 0)}`",
        f"- Matched-pair median token delta: `{summary.get('matched_pair_token_delta_median', 'n/a')}`",
        f"- Matched-pair median token delta %: `{summary.get('matched_pair_token_delta_pct_median', 'n/a')}`",
        f"- Claimable: `{summary.get('claimable', False)}`",
        "",
        "## Top Completion Failure Reasons",
        "",
    ]
    failures = summary.get("top_completion_failure_reasons", [])
    if failures:
        for item in failures:
            lines.append(f"- `{item['reason']}`: `{item['count']}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Advisory Warnings", ""])
    warnings = summary.get("advisory_warning_counts", {})
    if warnings:
        for key in sorted(warnings):
            lines.append(f"- `{key}`: `{warnings[key]}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Tool Engagement", ""])
    tool_stats = summary.get("tool_engagement_stats", {})
    for key in sorted(tool_stats):
        lines.append(f"- `{key}`: `{tool_stats[key]}`")
    if not tool_stats:
        lines.append("- none")

    turn_stats = summary.get("turn_diagnostics", {})
    if turn_stats:
        lines.extend(["", "## Turn Diagnostics", ""])
        for condition in ("baseline", "tok-universal"):
            condition_stats = turn_stats.get(condition, {})
            if not condition_stats:
                continue
            lines.append(f"- `{condition}` avg turns: `{condition_stats.get('avg_turns', 0.0)}`")
            lines.append(f"- `{condition}` avg tokens/turn: `{condition_stats.get('avg_tokens_per_turn', 0.0)}`")
        lines.append(f"- first cumulative parity turn: `{turn_stats.get('first_cumulative_parity_turn', 'none')}`")
        step_delta_pct = turn_stats.get("median_step_token_delta_pct", {})
        if step_delta_pct:
            top_steps = sorted(step_delta_pct.items(), key=lambda item: abs(float(item[1])), reverse=True)[:5]
            lines.append("- largest median step token delta %:")
            for step, value in top_steps:
                lines.append(f"  - step `{step}`: `{value}%`")
        runtime_warnings = turn_stats.get("runtime_contract_warnings", {})
        if runtime_warnings:
            lines.append("- runtime warnings:")
            for key in sorted(runtime_warnings):
                lines.append(f"  - `{key}`: `{runtime_warnings[key]}`")
        top_deltas = turn_stats.get("top_token_delta_tasks", [])
        if top_deltas:
            lines.append("- top token delta tasks:")
            for item in top_deltas[:5]:
                lines.append(
                    f"  - `{item['task_id']}` r{item['repeat_index']}: "
                    f"delta_tokens=`{item['delta_tokens']}`, delta_turns=`{item['delta_turns']}`"
                )
    lines.append("")
    return "\n".join(lines)


def render_tiny_patch_markdown(summary: dict[str, Any]) -> str:
    return render_patch_suite_markdown(summary, benchmark_name="tiny-patch")


def run_patch_suite_benchmark(
    *,
    benchmark_name: str,
    runner: LiveBenchmarkRunner,
    output_root: Path,
    repeats: int,
    repo_root: Path,
) -> tuple[CatalogBenchmarkRun, dict[str, Any], str]:
    catalog = load_patch_suite_catalog(benchmark_name=benchmark_name, repo_root=repo_root)
    catalog_run = run_catalog_benchmark_suite(
        catalog=catalog,
        lane_id="production_claude_lane",
        output_root=output_root,
        repeats=max(1, repeats),
        families=("execution_patch",),
        include_advisory=False,
        public_release_only=False,
        local_debug=True,
        runner=runner,
        repo_root=repo_root,
    )
    summary = summarize_patch_suite_run(catalog_run, output_root=output_root, benchmark_name=benchmark_name)
    markdown = render_patch_suite_markdown(summary, benchmark_name=benchmark_name)
    return catalog_run, summary, markdown


def run_tiny_patch_benchmark(
    *,
    runner: LiveBenchmarkRunner,
    output_root: Path,
    repeats: int,
    repo_root: Path,
) -> tuple[CatalogBenchmarkRun, dict[str, Any], str]:
    return run_patch_suite_benchmark(
        benchmark_name="tiny-patch",
        runner=runner,
        output_root=output_root,
        repeats=repeats,
        repo_root=repo_root,
    )
