from __future__ import annotations

import json
from pathlib import Path

from tok.testing.benchmark_suite import (
    BenchmarkCatalog,
    BenchmarkComparisonRun,
    BenchmarkLane,
    BenchmarkTaskManifest,
    build_benchmark_report,
)
from tok.testing.live_benchmark import LiveBenchmarkRunner

from ._evaluator import FamilyEvaluator
from ._loop_executor import ToolLoopExecutor
from ._materializer import TaskMaterializer
from ._models import (
    BenchmarkTaskRunResult,
    CatalogBenchmarkRun,
    TaskEvaluationResult,
)


def select_catalog_tasks(
    catalog: BenchmarkCatalog,
    *,
    families: tuple[str, ...],
    task_ids: tuple[str, ...],
    include_advisory: bool,
    public_release_only: bool,
) -> tuple[BenchmarkTaskManifest, ...]:
    requested_ids = set(task_ids)
    selected: list[BenchmarkTaskManifest] = []
    for task in catalog.tasks:
        if families and task.family not in families:
            continue
        if requested_ids and task.id not in requested_ids:
            continue
        if task.family == "real_session" and not include_advisory:
            continue
        if public_release_only and not task.public_release:
            continue
        selected.append(task)
    return tuple(selected)


def _local_failure_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "asset lock hash mismatch" in message:
        return "asset_lock_hash_mismatch"
    if "asset lock missing" in message or "workspace_sha256" in message:
        return "asset_lock_missing"
    if "workspace source not found" in message:
        return "workspace_source_missing"
    return "task_materialization_failed"


def _execution_failure_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "object has no attribute 'strip'" in message and "list" in message:
        return "adapter_payload_contract_error"
    if 'object has no attribute "strip"' in message and "list" in message:
        return "adapter_payload_contract_error"
    return "task_execution_failed"


def _materialization_failure_result(
    *,
    task: BenchmarkTaskManifest,
    lane: BenchmarkLane,
    repeat_index: int,
    condition: str,
    output_root: Path,
    error: Exception,
) -> BenchmarkTaskRunResult:
    output_root.mkdir(parents=True, exist_ok=True)
    details = {
        "failure_stage": "materialize",
        "failure_error": str(error),
    }
    evaluation = TaskEvaluationResult(
        success=False,
        grounding_success=False,
        details=details,
        notes=("materialization_failed",),
    )
    result = BenchmarkTaskRunResult(
        lane_id=lane.id,
        condition=condition,
        task_id=task.id,
        family=task.family,
        repeat_index=repeat_index,
        workspace_root=str(output_root / "workspace"),
        answer_text="",
        raw_response="",
        provider_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0},
        tool_calls=0,
        invalid_tool_calls=0,
        reacquisition_events=0,
        clean_exit=False,
        modified_files=tuple(),
        tool_records=tuple(),
        turns=tuple(),
        evaluation=evaluation,
        local_failure=_local_failure_code(error),
        notes=("materialization_failed",),
    )
    (output_root / "run.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


def _execution_failure_result(
    *,
    task: BenchmarkTaskManifest,
    lane: BenchmarkLane,
    repeat_index: int,
    condition: str,
    output_root: Path,
    error: Exception,
) -> BenchmarkTaskRunResult:
    output_root.mkdir(parents=True, exist_ok=True)
    details = {
        "failure_stage": "execution",
        "failure_error": str(error),
    }
    evaluation = TaskEvaluationResult(
        success=False,
        grounding_success=False,
        details=details,
        notes=("execution_failed",),
    )
    result = BenchmarkTaskRunResult(
        lane_id=lane.id,
        condition=condition,
        task_id=task.id,
        family=task.family,
        repeat_index=repeat_index,
        workspace_root=str(output_root / "workspace"),
        answer_text="",
        raw_response="",
        provider_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0},
        tool_calls=0,
        invalid_tool_calls=0,
        reacquisition_events=0,
        clean_exit=False,
        modified_files=tuple(),
        tool_records=tuple(),
        turns=tuple(),
        evaluation=evaluation,
        local_failure=_execution_failure_code(error),
        notes=("execution_failed",),
    )
    (output_root / "run.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


def run_catalog_benchmark_suite(
    *,
    catalog: BenchmarkCatalog,
    lane_id: str,
    output_root: Path,
    repeats: int,
    families: tuple[str, ...],
    task_ids: tuple[str, ...] = (),
    include_advisory: bool = False,
    public_release_only: bool = False,
    local_debug: bool = False,
    runner: LiveBenchmarkRunner,
    repo_root: Path | None = None,
    candidate_conditions: tuple[str, ...] = ("tok-universal",),
) -> CatalogBenchmarkRun:
    lane = catalog.lane_by_id(lane_id)
    tasks = select_catalog_tasks(
        catalog,
        families=families,
        task_ids=task_ids,
        include_advisory=include_advisory,
        public_release_only=public_release_only,
    )
    materializer = TaskMaterializer(catalog_root=Path(catalog.root), repo_root=repo_root)
    catalog_root = Path(catalog.root)
    loop_executor = ToolLoopExecutor(runner, catalog_root=catalog_root)
    evaluator = FamilyEvaluator(catalog_root=catalog_root)
    comparison_runs: list[BenchmarkComparisonRun] = []
    selected_task_ids = tuple(task.id for task in tasks)
    unique_candidates: list[str] = []
    for condition in candidate_conditions:
        normalized = str(condition or "").strip()
        if not normalized:
            continue
        if normalized == "baseline":
            msg = "candidate_conditions must not include baseline"
            raise ValueError(msg)
        if normalized in unique_candidates:
            continue
        unique_candidates.append(normalized)
    if not unique_candidates:
        msg = "candidate_conditions must include at least one condition"
        raise ValueError(msg)
    primary_candidate = "tok-universal" if "tok-universal" in unique_candidates else unique_candidates[0]
    condition_order = ("baseline", *tuple(unique_candidates))

    output_root.mkdir(parents=True, exist_ok=True)
    evaluator.validate_execution_evaluators(tasks)
    for task in tasks:
        for repeat_index in range(1, max(1, repeats) + 1):
            pair_results: dict[str, BenchmarkTaskRunResult] = {}
            for condition in condition_order:
                task_output = output_root / "tasks" / task.id / f"repeat_{repeat_index}" / condition
                try:
                    materialized = materializer.materialize(
                        task,
                        lane,
                        repeat_index=repeat_index,
                        condition=condition,
                        output_root=task_output,
                        reportable=not local_debug,
                        local_debug=local_debug,
                    )
                except Exception as exc:
                    pair_results[condition] = _materialization_failure_result(
                        task=task,
                        lane=lane,
                        repeat_index=repeat_index,
                        condition=condition,
                        output_root=task_output,
                        error=exc,
                    )
                    continue
                try:
                    pair_results[condition] = loop_executor.run_task(
                        materialized,
                        output_root=task_output,
                    )
                except Exception as exc:
                    pair_results[condition] = _execution_failure_result(
                        task=task,
                        lane=lane,
                        repeat_index=repeat_index,
                        condition=condition,
                        output_root=task_output,
                        error=exc,
                    )
            comparison = evaluator.compare_pair(
                task=task,
                lane_id=lane.id,
                repeat_index=repeat_index,
                baseline=pair_results["baseline"],
                candidate=pair_results[primary_candidate],
            )
            comparison_runs.append(comparison)
            compare_path = output_root / "tasks" / task.id / f"repeat_{repeat_index}" / "compare.json"
            compare_path.write_text(json.dumps(comparison.to_dict(), indent=2))
            for condition in unique_candidates:
                if condition == primary_candidate:
                    continue
                extra_comparison = evaluator.compare_pair(
                    task=task,
                    lane_id=lane.id,
                    repeat_index=repeat_index,
                    baseline=pair_results["baseline"],
                    candidate=pair_results[condition],
                )
                extra_compare_path = (
                    output_root / "tasks" / task.id / f"repeat_{repeat_index}" / f"compare_{condition}.json"
                )
                extra_compare_path.write_text(json.dumps(extra_comparison.to_dict(), indent=2))

    report = build_benchmark_report(
        catalog,
        comparison_runs,
        title="Production Tok Benchmark Report",
        notes=(
            "catalog_executor",
            f"lane={lane.id}",
            f"public_release_only={public_release_only}",
        ),
    )
    raw_runs_path = output_root / "raw_runs.json"
    raw_runs_path.write_text(
        json.dumps(
            {
                "title": "Production Tok Benchmark Report",
                "runs": [run.to_dict() for run in comparison_runs],
            },
            indent=2,
        )
    )
    (output_root / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    return CatalogBenchmarkRun(
        lane_id=lane.id,
        selected_task_ids=selected_task_ids,
        runs=tuple(comparison_runs),
        report=report,
    )


def render_combined_benchmark_summary(
    *,
    legacy_benchmarks: tuple[str, ...],
    catalog_run: CatalogBenchmarkRun,
    catalog_report_markdown: str,
) -> str:
    lines = [
        "# Live Benchmark Summary",
        "",
        "## Replay Stability",
        "",
    ]
    for benchmark in legacy_benchmarks:
        lines.append(f"- `{benchmark}` written under `replay/`")
    lines.extend(
        [
            "",
            "## Catalog Benchmark",
            "",
            catalog_report_markdown.strip(),
            "",
        ]
    )
    if catalog_run.selected_task_ids:
        lines.append(
            "- Catalog task coverage: " + ", ".join(f"`{task_id}`" for task_id in catalog_run.selected_task_ids)
        )
        lines.append("")
    return "\n".join(lines)
