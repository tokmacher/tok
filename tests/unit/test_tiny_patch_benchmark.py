from __future__ import annotations

import json
from types import SimpleNamespace

from tok.testing.benchmark_suite import BenchmarkComparisonRun
from tok.testing.tiny_patch_benchmark import load_patch_suite_catalog, load_tiny_patch_catalog, summarize_tiny_patch_run


def _run(**overrides: object) -> BenchmarkComparisonRun:
    payload: dict[str, object] = {
        "lane_id": "production_claude_lane",
        "task_id": "tiny.calc.add",
        "family": "execution_patch",
        "repeat_index": 1,
        "public_release": False,
        "baseline_success": True,
        "tok_success": True,
        "quality_gate_passed": True,
        "total_token_delta": -12,
        "latency_delta_ms": 1.2,
        "reacquisition_events": 0,
        "invalid_tool_calls": 0,
        "paired_result_stable": True,
        "baseline_grounding_success": True,
        "tok_grounding_success": True,
        "baseline_tool_calls": 2,
        "tok_tool_calls": 2,
        "format_contract_violations": [],
        "tool_engagement_stats": {},
        "matched_completion_pair": True,
    }
    payload.update(overrides)
    return BenchmarkComparisonRun.from_dict(payload)


def test_load_tiny_patch_catalog_contains_five_execution_tasks(tmp_path) -> None:
    catalog = load_tiny_patch_catalog(repo_root=tmp_path)
    assert len(catalog.tasks) == 5
    assert all(task.family == "execution_patch" for task in catalog.tasks)
    assert all(task.workspace_source.get("kind") == "local_checkout" for task in catalog.tasks)


def test_load_small_medium_and_large_patch_catalogs(tmp_path) -> None:
    small = load_patch_suite_catalog(benchmark_name="small-patch", repo_root=tmp_path)
    medium = load_patch_suite_catalog(benchmark_name="medium-patch", repo_root=tmp_path)
    large = load_patch_suite_catalog(benchmark_name="large-patch", repo_root=tmp_path)
    assert len(small.tasks) == 3
    assert len(medium.tasks) == 3
    assert len(large.tasks) == 3
    assert all(task.family == "execution_patch" for task in (*small.tasks, *medium.tasks, *large.tasks))


def test_summarize_tiny_patch_run_uses_matched_completion_pairs_and_failure_notes(tmp_path) -> None:
    tasks_root = tmp_path / "tasks" / "tiny.calc.add" / "repeat_1"
    (tasks_root / "baseline").mkdir(parents=True)
    (tasks_root / "tok-universal").mkdir(parents=True)
    (tasks_root / "baseline" / "run.json").write_text(
        json.dumps({"evaluation": {"success": False, "notes": ["hidden_tests_failed"]}})
    )
    (tasks_root / "tok-universal" / "run.json").write_text(
        json.dumps({"evaluation": {"success": False, "notes": ["hidden_tests_failed", "allowed_path_check_failed"]}})
    )

    catalog_like = SimpleNamespace(
        runs=(
            _run(task_id="tiny.calc.add", repeat_index=1, total_token_delta=-10, matched_completion_pair=True),
            _run(
                task_id="tiny.calc.subtract",
                repeat_index=1,
                baseline_success=False,
                tok_success=False,
                quality_gate_passed=False,
                total_token_delta=5,
                matched_completion_pair=False,
                invalid_tool_calls=1,
                tok_tool_calls=0,
                format_contract_violations=["evidence_block_count"],
            ),
        )
    )

    summary = summarize_tiny_patch_run(catalog_like, output_root=tmp_path)

    assert summary["runs"] == 2
    assert summary["completion_success_rate"]["baseline"] == 0.5
    assert summary["completion_success_rate"]["tok-universal"] == 0.5
    assert summary["matched_completion_pair_count"] == 1
    assert summary["matched_pair_token_delta_median"] == -10.0
    assert summary["advisory_warning_counts"]["invalid_tool_calls_present"] == 1
    assert summary["advisory_warning_counts"]["tok_zero_tool_calls"] == 1
    reasons = {item["reason"]: item["count"] for item in summary["top_completion_failure_reasons"]}
    assert reasons["baseline:hidden_tests_failed"] == 1
    assert reasons["tok-universal:hidden_tests_failed"] == 1
