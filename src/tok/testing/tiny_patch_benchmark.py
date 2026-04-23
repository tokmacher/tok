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
_TINY_MIN_MATCHED_COMPLETION_PAIRS = 15
_CONTROLLED_TOK_CONDITION = "tok-controlled"


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
        "allowed_tools": ["list_dir", "view_file", "grep_search", "edit_file", "run_tests", "bash"],
        "time_budget_minutes": time_budget_minutes,
        "step_budget": step_budget,
        "success_evaluator": {
            "kind": "execution_patch",
            "clean_exit_required": False,
            "expect_initial_hidden_failure": True,
        },
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


def _condition_order(tasks_root: Path) -> tuple[str, ...]:
    conditions: set[str] = set()
    if tasks_root.exists():
        for run_path in tasks_root.glob("*/repeat_*/*/run.json"):
            condition = run_path.parent.name.strip()
            if condition:
                conditions.add(condition)
    ordered: list[str] = []
    for preferred in ("baseline", "tok-universal", _CONTROLLED_TOK_CONDITION):
        if preferred in conditions:
            ordered.append(preferred)
            conditions.remove(preferred)
    ordered.extend(sorted(conditions))
    return tuple(ordered)


def _collect_completion_failure_reasons(tasks_root: Path) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    if not tasks_root.exists():
        return []
    for condition in _condition_order(tasks_root):
        for run_path in tasks_root.glob(f"*/repeat_*/{condition}/run.json"):
            payload = json.loads(run_path.read_text())
            if bool(payload.get("evaluation", {}).get("success")):
                continue
            for note in payload.get("evaluation", {}).get("notes", []):
                counts[f"{condition}:{note}"] += 1
    return [{"reason": reason, "count": count} for reason, count in counts.most_common(10)]


def _load_condition_runs(tasks_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for condition in _condition_order(tasks_root):
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

    condition_order = _condition_order(task_runs_root)
    by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in condition_order}
    step_totals: dict[str, dict[int, list[int]]] = {condition: defaultdict(list) for condition in condition_order}
    tok_signal_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    tool_protocol_retry_reason_counts: Counter[str] = Counter()
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
            diagnostics = turn.get("diagnostics") or {}
            retry_count = int(diagnostics.get("tool_protocol_retry_count", 0) or 0)
            retry_success = int(diagnostics.get("tool_protocol_retry_success", 0) or 0)
            retry_reason = str(diagnostics.get("tool_protocol_retry_reason", "") or "").strip()
            if retry_count > 0:
                warning_counts["tool_protocol_retry_count"] += retry_count
                if retry_success > 0:
                    warning_counts["tool_protocol_retry_success"] += retry_success
                else:
                    warning_counts["tool_protocol_retry_failure"] += retry_count
                if retry_reason:
                    tool_protocol_retry_reason_counts[retry_reason] += 1
            if condition != "tok-universal":
                continue
            signals = (turn.get("response_metrics") or {}).get("response_behavior_signals") or {}
            for key, value in signals.items():
                if isinstance(value, int | float) and value:
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

    per_step_medians: dict[str, dict[str, float]] = {condition: {} for condition in condition_order}
    for condition in condition_order:
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
    baseline_steps = step_totals.get("baseline", {})
    tok_steps = step_totals.get("tok-universal", {})
    all_steps = sorted({*baseline_steps.keys(), *tok_steps.keys()})
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
        "baseline": _averages(by_condition.get("baseline", [])),
        "tok-universal": _averages(by_condition.get("tok-universal", [])),
        "median_step_total_tokens": per_step_medians,
        "median_step_token_delta": step_delta_tokens,
        "median_step_token_delta_pct": step_delta_pct,
        "cumulative_median_step_token_delta": cumulative_step_delta_tokens,
        "first_cumulative_parity_turn": first_cumulative_parity_turn,
        "top_tok_response_signals": [
            {"signal": signal, "count": count} for signal, count in tok_signal_counts.most_common(12)
        ],
        "runtime_contract_warnings": dict(warning_counts),
        "tool_protocol_retry_reasons": dict(tool_protocol_retry_reason_counts),
        "top_token_delta_tasks": top_delta_tasks[:10],
    }


def _extract_run_success(payload: dict[str, Any]) -> bool:
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict) and "success" in evaluation:
        return bool(evaluation.get("success"))
    return bool(payload.get("success"))


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 1)
    ordered = sorted(float(item) for item in values)
    index = int(round((len(ordered) - 1) * pct))
    index = max(0, min(index, len(ordered) - 1))
    return round(float(ordered[index]), 1)


def _build_forensics_report(task_runs_root: Path) -> dict[str, Any]:
    loaded_runs = _load_condition_runs(task_runs_root)
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {
        (item["task_id"], item["repeat_index"], item["condition"]): item["payload"] for item in loaded_runs
    }
    keys = sorted({(item["task_id"], item["repeat_index"]) for item in loaded_runs})
    candidates = tuple(condition for condition in _condition_order(task_runs_root) if condition != "baseline")
    per_run: list[dict[str, Any]] = []
    comparator_summaries: dict[str, dict[str, Any]] = {}

    for condition in candidates:
        rows: list[dict[str, Any]] = []
        for task_id, repeat_index in keys:
            baseline_payload = by_key.get((task_id, repeat_index, "baseline"))
            candidate_payload = by_key.get((task_id, repeat_index, condition))
            if not isinstance(baseline_payload, dict) or not isinstance(candidate_payload, dict):
                continue
            baseline_usage = baseline_payload.get("provider_usage") or {}
            candidate_usage = candidate_payload.get("provider_usage") or {}
            baseline_total = int(baseline_usage.get("total_tokens", 0) or 0)
            candidate_total = int(candidate_usage.get("total_tokens", 0) or 0)
            baseline_prompt = int(baseline_usage.get("prompt_tokens", 0) or 0)
            candidate_prompt = int(candidate_usage.get("prompt_tokens", 0) or 0)
            baseline_completion = int(baseline_usage.get("completion_tokens", 0) or 0)
            candidate_completion = int(candidate_usage.get("completion_tokens", 0) or 0)
            baseline_tool_calls = int(baseline_payload.get("tool_calls", 0) or 0)
            candidate_tool_calls = int(candidate_payload.get("tool_calls", 0) or 0)
            baseline_turns = len(baseline_payload.get("turns") or [])
            candidate_turns = len(candidate_payload.get("turns") or [])
            baseline_success = _extract_run_success(baseline_payload)
            candidate_success = _extract_run_success(candidate_payload)

            saved_total = 0
            codec_saved: Counter[str] = Counter()
            behavior_counters: Counter[str] = Counter()
            retry_reason_counts: Counter[str] = Counter()
            for turn in candidate_payload.get("turns") or []:
                compression_metrics = turn.get("compression_metrics") or {}
                response_metrics = turn.get("response_metrics") or {}
                turn_diagnostics = turn.get("diagnostics") or {}
                saved_total += int(compression_metrics.get("total_saved_tokens", 0) or 0)
                type_breakdown = compression_metrics.get("type_breakdown") or {}
                for key, value in type_breakdown.items():
                    if isinstance(value, int | float) and value:
                        codec_saved[str(key)] += int(value)
                input_signals = compression_metrics.get("input_behavior_signals") or {}
                response_signals = response_metrics.get("response_behavior_signals") or {}
                for signal_name in (
                    "request_policy_tool_compatible",
                    "tok_history_pairing_safety_degraded",
                    "tok_history_compression_skipped",
                ):
                    behavior_counters[signal_name] += int(input_signals.get(signal_name, 0) or 0)
                    behavior_counters[signal_name] += int(response_signals.get(signal_name, 0) or 0)
                behavior_counters["tool_protocol_retry_count"] += int(
                    turn_diagnostics.get("tool_protocol_retry_count", 0) or 0
                )
                behavior_counters["tool_protocol_retry_success"] += int(
                    turn_diagnostics.get("tool_protocol_retry_success", 0) or 0
                )
                retry_reason = str(turn_diagnostics.get("tool_protocol_retry_reason", "") or "").strip()
                if retry_reason:
                    retry_reason_counts[retry_reason] += 1

            row = {
                "task_id": task_id,
                "repeat_index": repeat_index,
                "candidate_condition": condition,
                "baseline_success": baseline_success,
                "candidate_success": candidate_success,
                "matched_success_pair": bool(baseline_success and candidate_success),
                "token_delta_total": candidate_total - baseline_total,
                "token_delta_prompt": candidate_prompt - baseline_prompt,
                "token_delta_completion": candidate_completion - baseline_completion,
                "turn_delta": candidate_turns - baseline_turns,
                "tool_call_delta": candidate_tool_calls - baseline_tool_calls,
                "compression_recovery_estimate": int(saved_total),
                "behavior_inflation_estimate": int((candidate_total - baseline_total) + saved_total),
                "compression_total_saved_tokens": int(saved_total),
                "codec_family_saved_tokens": dict(codec_saved),
                "behavior_counters": dict(behavior_counters),
                "tool_protocol_retry_reason_counts": dict(retry_reason_counts),
            }
            rows.append(row)
            per_run.append(row)

        token_deltas = [float(row["token_delta_total"]) for row in rows]
        matched_deltas = [float(row["token_delta_total"]) for row in rows if row["matched_success_pair"]]
        turn_deltas = [float(row["turn_delta"]) for row in rows]
        tool_deltas = [float(row["tool_call_delta"]) for row in rows]
        baseline_successes = [1.0 if row["baseline_success"] else 0.0 for row in rows]
        candidate_successes = [1.0 if row["candidate_success"] else 0.0 for row in rows]
        compression_total = int(sum(int(row["compression_total_saved_tokens"]) for row in rows))
        behavior_total = int(sum(int(row["behavior_inflation_estimate"]) for row in rows))

        codec_totals: Counter[str] = Counter()
        behavior_totals: Counter[str] = Counter()
        retry_reason_totals: Counter[str] = Counter()
        for row in rows:
            for key, value in (row.get("codec_family_saved_tokens") or {}).items():
                codec_totals[str(key)] += int(value or 0)
            for key, value in (row.get("behavior_counters") or {}).items():
                behavior_totals[str(key)] += int(value or 0)
            for key, value in (row.get("tool_protocol_retry_reason_counts") or {}).items():
                retry_reason_totals[str(key)] += int(value or 0)
        outliers = sorted(rows, key=lambda row: abs(int(row["token_delta_total"])), reverse=True)[:10]
        comparator_summaries[condition] = {
            "runs": len(rows),
            "completion_success_rate": {
                "baseline": round(sum(baseline_successes) / len(baseline_successes), 3) if baseline_successes else 0.0,
                "candidate": (
                    round(sum(candidate_successes) / len(candidate_successes), 3) if candidate_successes else 0.0
                ),
            },
            "matched_success_pair_count": sum(1 for row in rows if row["matched_success_pair"]),
            "token_delta_total_sum": int(sum(token_deltas)) if token_deltas else 0,
            "token_delta_total_median": round(float(statistics.median(token_deltas)), 1) if token_deltas else 0.0,
            "token_delta_total_p90": _percentile(token_deltas, 0.9),
            "matched_token_delta_median": round(float(statistics.median(matched_deltas)), 1)
            if matched_deltas
            else None,
            "turn_delta_median": round(float(statistics.median(turn_deltas)), 1) if turn_deltas else 0.0,
            "turn_delta_p90": _percentile(turn_deltas, 0.9),
            "tool_call_delta_median": round(float(statistics.median(tool_deltas)), 1) if tool_deltas else 0.0,
            "tool_call_delta_p90": _percentile(tool_deltas, 0.9),
            "compression_recovery_estimate_total": compression_total,
            "behavior_inflation_estimate_total": behavior_total,
            "codec_family_saved_tokens": dict(codec_totals),
            "behavior_counters": dict(behavior_totals),
            "tool_protocol_retry_reason_counts": dict(retry_reason_totals),
            "top_token_delta_outliers": [
                {
                    "task_id": row["task_id"],
                    "repeat_index": row["repeat_index"],
                    "token_delta_total": row["token_delta_total"],
                    "turn_delta": row["turn_delta"],
                    "tool_call_delta": row["tool_call_delta"],
                }
                for row in outliers
            ],
        }

    return {
        "candidates": comparator_summaries,
        "per_run_attribution": sorted(
            per_run,
            key=lambda row: (str(row["task_id"]), int(row["repeat_index"]), str(row["candidate_condition"])),
        ),
    }


def _matched_pair_threshold(*, benchmark_name: str) -> int:
    if benchmark_name == "tiny-patch":
        return _TINY_MIN_MATCHED_COMPLETION_PAIRS
    return 1


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
            "matched_pair_token_delta_pct_median": None,
            "claimable": False,
            "token_cost_decision_grade": False,
            "token_cost_decision_reasons": ["no_runs_available"],
            "token_cost_decision_gate": {
                "matched_pair_threshold": _matched_pair_threshold(benchmark_name=benchmark_name),
                "matched_pair_count": 0,
                "passed": False,
            },
            "top_completion_failure_reasons": [],
            "advisory_warning_counts": {},
            "tool_engagement_stats": {},
            "fairness_diagnostics": {
                "execution_contract_compliance_rate": {"baseline": 0.0, "tok-universal": 0.0},
                "loop_recovery_trigger_counts": {"baseline": 0, "tok-universal": 0},
                "premature_final_counts": {"baseline": 0, "tok-universal": 0},
                "tool_required_latch_active_counts": {"baseline": 0, "tok-universal": 0},
                "integrity_clean_matched_pair_count": 0,
                "integrity_blocked_matched_pair_count": 0,
                "matched_pair_blocker_counts": {},
                "matched_pair_blocker_rates": {},
                "matched_pair_blocked_by_asymmetry_count": 0,
                "matched_pair_blocked_by_artifact_count": 0,
                "matched_pair_asymmetry_non_blocking_count": 0,
                "matched_pair_blocked_by_asymmetry_pct": 0.0,
                "matched_pair_blocked_by_artifact_pct": 0.0,
            },
            "comparators": {},
            "attribution": {},
            "stability": {},
        }

    baseline_rate = round(sum(1 for run in runs if run.baseline_success) / len(runs), 3)
    tok_rate = round(sum(1 for run in runs if run.tok_success) / len(runs), 3)
    matched_deltas = [run.total_token_delta for run in runs if run.matched_completion_pair]
    task_runs_root = output_root / "tasks"
    loaded_condition_runs = _load_condition_runs(task_runs_root)
    turn_diagnostics = _turn_diagnostics(task_runs_root=task_runs_root, comparison_runs=runs)
    forensics_report = _build_forensics_report(task_runs_root)
    comparator_summaries = dict(forensics_report.get("candidates") or {})
    baseline_totals = {
        (item["task_id"], item["repeat_index"], item["condition"]): int(
            (item["payload"].get("provider_usage") or {}).get("total_tokens", 0) or 0
        )
        for item in loaded_condition_runs
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
    matched_pair_threshold = _matched_pair_threshold(benchmark_name=benchmark_name)
    matched_pair_count = len(matched_deltas)
    matched_pair_gate_passed = matched_pair_count >= matched_pair_threshold
    integrity_clean_matched_pair_count = sum(
        1
        for run in runs
        if run.matched_completion_pair and bool((run.tool_engagement_stats or {}).get("decision_grade", True))
    )
    integrity_blocked_matched_pair_count = matched_pair_count - integrity_clean_matched_pair_count
    token_cost_decision_grade = (
        claimable and matched_pair_gate_passed and integrity_blocked_matched_pair_count == 0 and matched_pair_count > 0
    )
    token_cost_decision_reasons: list[str] = []
    if not matched_pair_gate_passed:
        token_cost_decision_reasons.append("matched_completion_pairs_below_threshold")
    if integrity_blocked_matched_pair_count > 0:
        token_cost_decision_reasons.append("integrity_flags_present_in_matched_pairs")
    if not claimable:
        token_cost_decision_reasons.append("claimability_gate_not_met")

    execution_contract_counts: dict[str, dict[str, int]] = {
        condition: {"met": 0, "total": 0} for condition in _condition_order(task_runs_root)
    }
    loop_recovery_trigger_counts: Counter[str] = Counter()
    premature_final_counts: Counter[str] = Counter()
    tool_required_latch_active_counts: Counter[str] = Counter()
    matched_pair_blocker_counts: Counter[str] = Counter()
    matched_pair_blocked_by_asymmetry_count = 0
    matched_pair_blocked_by_artifact_count = 0
    matched_pair_asymmetry_non_blocking_count = 0

    for run in runs:
        stats = run.tool_engagement_stats or {}
        premature_final_counts["baseline"] += int(stats.get("baseline_premature_final_count", 0) or 0)
        premature_final_counts["tok-universal"] += int(stats.get("tok_premature_final_count", 0) or 0)
        tool_required_latch_active_counts["baseline"] += int(
            stats.get("baseline_tool_required_latch_active_count", 0) or 0
        )
        tool_required_latch_active_counts["tok-universal"] += int(
            stats.get("tok_tool_required_latch_active_count", 0) or 0
        )

    for run in runs:
        if not run.matched_completion_pair:
            continue
        stats = run.tool_engagement_stats or {}
        if bool(stats.get("decision_grade", True)):
            continue
        blocker_keys: set[str] = set()
        artifact_flags = tuple(stats.get("integrity_artifact_flags") or ())
        decision_blockers = tuple(stats.get("decision_grade_blockers") or ())
        asymmetry_flags = tuple(stats.get("integrity_asymmetry_flags") or ())
        blocker_keys.update(str(item) for item in artifact_flags if isinstance(item, str) and item.strip())
        blocker_keys.update(str(item) for item in decision_blockers if isinstance(item, str) and item.strip())
        if not blocker_keys:
            blocker_keys.add("unknown_decision_grade_blocker")
        for blocker in blocker_keys:
            matched_pair_blocker_counts[blocker] += 1
        if artifact_flags:
            matched_pair_blocked_by_artifact_count += 1
        if decision_blockers:
            matched_pair_blocked_by_asymmetry_count += 1
        elif asymmetry_flags:
            matched_pair_asymmetry_non_blocking_count += 1

    for item in loaded_condition_runs:
        condition = item["condition"]
        payload = item["payload"] if isinstance(item["payload"], dict) else {}
        details = (payload.get("evaluation") or {}).get("details") or {}
        if "execution_contract_met" in details:
            execution_contract_counts.setdefault(condition, {"met": 0, "total": 0})
            execution_contract_counts[condition]["total"] += 1
            if bool(details.get("execution_contract_met")):
                execution_contract_counts[condition]["met"] += 1
        for note in payload.get("notes") or []:
            if isinstance(note, str) and note.startswith("read_only_loop_recovery_step_"):
                loop_recovery_trigger_counts[condition] += 1

    execution_contract_compliance_rate = {
        condition: (round(counts["met"] / counts["total"], 3) if counts["total"] > 0 else 0.0)
        for condition, counts in execution_contract_counts.items()
    }
    matched_pair_blocker_rates = {
        key: (round(float(count) / float(matched_pair_count), 3) if matched_pair_count > 0 else 0.0)
        for key, count in sorted(matched_pair_blocker_counts.items())
    }

    return {
        "benchmark": benchmark_name,
        "runs": len(runs),
        "completion_success_rate": {"baseline": baseline_rate, "tok-universal": tok_rate},
        "matched_completion_pair_count": matched_pair_count,
        "matched_pair_token_delta_median": (
            round(float(statistics.median(matched_deltas)), 1) if matched_deltas else None
        ),
        "matched_pair_token_delta_pct_median": (
            round(float(statistics.median(matched_delta_pct)), 1) if matched_delta_pct else None
        ),
        "claimable": claimable,
        "token_cost_decision_grade": token_cost_decision_grade,
        "token_cost_decision_reasons": token_cost_decision_reasons,
        "token_cost_decision_gate": {
            "matched_pair_threshold": matched_pair_threshold,
            "matched_pair_count": matched_pair_count,
            "passed": matched_pair_gate_passed,
        },
        "top_completion_failure_reasons": _collect_completion_failure_reasons(task_runs_root),
        "advisory_warning_counts": dict(advisory_counts),
        "tool_engagement_stats": {
            "baseline_avg_tool_calls": round(sum(baseline_tool_calls) / len(baseline_tool_calls), 2),
            "tok_avg_tool_calls": round(sum(tok_tool_calls) / len(tok_tool_calls), 2),
        },
        "fairness_diagnostics": {
            "execution_contract_compliance_rate": execution_contract_compliance_rate,
            "loop_recovery_trigger_counts": {
                "baseline": int(loop_recovery_trigger_counts["baseline"]),
                "tok-universal": int(loop_recovery_trigger_counts["tok-universal"]),
            },
            "premature_final_counts": {
                "baseline": int(premature_final_counts["baseline"]),
                "tok-universal": int(premature_final_counts["tok-universal"]),
            },
            "tool_required_latch_active_counts": {
                "baseline": int(tool_required_latch_active_counts["baseline"]),
                "tok-universal": int(tool_required_latch_active_counts["tok-universal"]),
            },
            "integrity_clean_matched_pair_count": integrity_clean_matched_pair_count,
            "integrity_blocked_matched_pair_count": integrity_blocked_matched_pair_count,
            "matched_pair_blocker_counts": dict(sorted(matched_pair_blocker_counts.items())),
            "matched_pair_blocker_rates": matched_pair_blocker_rates,
            "matched_pair_blocked_by_asymmetry_count": matched_pair_blocked_by_asymmetry_count,
            "matched_pair_blocked_by_artifact_count": matched_pair_blocked_by_artifact_count,
            "matched_pair_asymmetry_non_blocking_count": matched_pair_asymmetry_non_blocking_count,
            "matched_pair_blocked_by_asymmetry_pct": (
                round((float(matched_pair_blocked_by_asymmetry_count) / float(matched_pair_count)) * 100.0, 1)
                if matched_pair_count > 0
                else 0.0
            ),
            "matched_pair_blocked_by_artifact_pct": (
                round((float(matched_pair_blocked_by_artifact_count) / float(matched_pair_count)) * 100.0, 1)
                if matched_pair_count > 0
                else 0.0
            ),
        },
        "turn_diagnostics": turn_diagnostics,
        "comparators": comparator_summaries,
        "attribution": {
            "tok-universal": comparator_summaries.get("tok-universal", {}),
            _CONTROLLED_TOK_CONDITION: comparator_summaries.get(_CONTROLLED_TOK_CONDITION, {}),
        },
        "stability": {
            condition: {
                "token_delta_total_p50": item.get("token_delta_total_median"),
                "token_delta_total_p90": item.get("token_delta_total_p90"),
                "turn_delta_p50": item.get("turn_delta_median"),
                "turn_delta_p90": item.get("turn_delta_p90"),
                "tool_call_delta_p50": item.get("tool_call_delta_median"),
                "tool_call_delta_p90": item.get("tool_call_delta_p90"),
                "top_token_delta_outliers": item.get("top_token_delta_outliers", []),
            }
            for condition, item in sorted(comparator_summaries.items())
        },
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
        f"- Token/cost decision grade: `{summary.get('token_cost_decision_grade', False)}`",
        "",
        "## Top Completion Failure Reasons",
        "",
    ]
    decision_gate = summary.get("token_cost_decision_gate", {})
    if decision_gate:
        lines.append(
            "- Token/cost gate: matched pairs "
            f"`{decision_gate.get('matched_pair_count', 0)}` / "
            f"`{decision_gate.get('matched_pair_threshold', 0)}` "
            f"(passed=`{decision_gate.get('passed', False)}`)"
        )
    decision_reasons = summary.get("token_cost_decision_reasons", [])
    if decision_reasons:
        lines.append("- Token/cost provisional reasons: " + ", ".join(f"`{reason}`" for reason in decision_reasons))
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

    fairness_stats = summary.get("fairness_diagnostics", {})
    if fairness_stats:
        lines.extend(["", "## Fairness Diagnostics", ""])
        contract_rates = fairness_stats.get("execution_contract_compliance_rate", {})
        for condition in ("baseline", "tok-universal"):
            if condition in contract_rates:
                lines.append(f"- `{condition}` execution-contract compliance: `{contract_rates.get(condition, 0.0)}`")
        recovery_counts = fairness_stats.get("loop_recovery_trigger_counts", {})
        for condition in ("baseline", "tok-universal"):
            if condition in recovery_counts:
                lines.append(f"- `{condition}` loop-recovery triggers: `{recovery_counts.get(condition, 0)}`")
        premature_counts = fairness_stats.get("premature_final_counts", {})
        for condition in ("baseline", "tok-universal"):
            if condition in premature_counts:
                lines.append(f"- `{condition}` premature-final count: `{premature_counts.get(condition, 0)}`")
        latch_counts = fairness_stats.get("tool_required_latch_active_counts", {})
        for condition in ("baseline", "tok-universal"):
            if condition in latch_counts:
                lines.append(f"- `{condition}` tool-required latch active count: `{latch_counts.get(condition, 0)}`")
        lines.append(
            f"- integrity-clean matched pairs: `{fairness_stats.get('integrity_clean_matched_pair_count', 0)}`"
        )
        lines.append(
            f"- integrity-blocked matched pairs: `{fairness_stats.get('integrity_blocked_matched_pair_count', 0)}`"
        )
        lines.append(
            "- matched-pair blocked by asymmetry: "
            f"`{fairness_stats.get('matched_pair_blocked_by_asymmetry_count', 0)}` "
            f"(`{fairness_stats.get('matched_pair_blocked_by_asymmetry_pct', 0.0)}%`)"
        )
        lines.append(
            "- matched-pair blocked by artifacts: "
            f"`{fairness_stats.get('matched_pair_blocked_by_artifact_count', 0)}` "
            f"(`{fairness_stats.get('matched_pair_blocked_by_artifact_pct', 0.0)}%`)"
        )
        lines.append(
            "- matched-pair asymmetry present but non-blocking: "
            f"`{fairness_stats.get('matched_pair_asymmetry_non_blocking_count', 0)}`"
        )
        blocker_counts = fairness_stats.get("matched_pair_blocker_counts", {})
        if blocker_counts:
            lines.append("- matched-pair blocker counts:")
            blocker_rates = fairness_stats.get("matched_pair_blocker_rates", {})
            for key in sorted(blocker_counts):
                rate = blocker_rates.get(key, 0.0)
                lines.append(f"  - `{key}`: `{blocker_counts[key]}` (`{rate}`)")

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
    comparators = summary.get("comparators", {})
    if comparators:
        lines.extend(["", "## Comparator Diagnostics", ""])
        for condition, stats in sorted(comparators.items()):
            lines.append(f"- `{condition}` runs: `{stats.get('runs', 0)}`")
            completion_stats = stats.get("completion_success_rate", {})
            lines.append(
                f"  - completion success baseline/candidate: "
                f"`{completion_stats.get('baseline', 0.0)}` / `{completion_stats.get('candidate', 0.0)}`"
            )
            lines.append(
                f"  - token delta sum/median/p90: "
                f"`{stats.get('token_delta_total_sum', 0)}` / "
                f"`{stats.get('token_delta_total_median', 0.0)}` / "
                f"`{stats.get('token_delta_total_p90', 0.0)}`"
            )
            lines.append(
                f"  - turn delta p50/p90: "
                f"`{stats.get('turn_delta_median', 0.0)}` / `{stats.get('turn_delta_p90', 0.0)}`"
            )
            lines.append(
                f"  - tool-call delta p50/p90: "
                f"`{stats.get('tool_call_delta_median', 0.0)}` / `{stats.get('tool_call_delta_p90', 0.0)}`"
            )
            lines.append(
                f"  - behavior/compression estimate totals: "
                f"`{stats.get('behavior_inflation_estimate_total', 0)}` / "
                f"`{stats.get('compression_recovery_estimate_total', 0)}`"
            )

    stability = summary.get("stability", {})
    if stability:
        lines.extend(["", "## Stability", ""])
        for condition, stats in sorted(stability.items()):
            lines.append(
                f"- `{condition}` token delta p50/p90: "
                f"`{stats.get('token_delta_total_p50', 0.0)}` / `{stats.get('token_delta_total_p90', 0.0)}`"
            )
            outliers = stats.get("top_token_delta_outliers", [])
            if outliers:
                lines.append("  - top outliers:")
                for item in outliers[:3]:
                    lines.append(
                        f"    - `{item['task_id']}` r{item['repeat_index']}: "
                        f"delta_tokens=`{item['token_delta_total']}`, "
                        f"delta_turns=`{item['turn_delta']}`, "
                        f"delta_tool_calls=`{item['tool_call_delta']}`"
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
        candidate_conditions=("tok-universal", _CONTROLLED_TOK_CONDITION),
    )
    forensics_report = _build_forensics_report(output_root / "tasks")
    (output_root / "attribution_report.json").write_text(json.dumps(forensics_report, indent=2))
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
