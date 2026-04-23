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
        "tool_engagement_stats": {"decision_grade": True},
        "matched_completion_pair": True,
    }
    payload.update(overrides)
    return BenchmarkComparisonRun.from_dict(payload)


def test_load_tiny_patch_catalog_contains_five_execution_tasks(tmp_path) -> None:
    catalog = load_tiny_patch_catalog(repo_root=tmp_path)
    assert len(catalog.tasks) == 5
    assert all(task.family == "execution_patch" for task in catalog.tasks)
    assert all(task.workspace_source.get("kind") == "local_checkout" for task in catalog.tasks)
    assert all(bool(task.success_evaluator.get("expect_initial_hidden_failure")) for task in catalog.tasks)


def test_load_small_medium_and_large_patch_catalogs(tmp_path) -> None:
    small = load_patch_suite_catalog(benchmark_name="small-patch", repo_root=tmp_path)
    medium = load_patch_suite_catalog(benchmark_name="medium-patch", repo_root=tmp_path)
    large = load_patch_suite_catalog(benchmark_name="large-patch", repo_root=tmp_path)
    assert len(small.tasks) == 3
    assert len(medium.tasks) == 3
    assert len(large.tasks) == 3
    assert all(task.family == "execution_patch" for task in (*small.tasks, *medium.tasks, *large.tasks))


def test_summarize_tiny_patch_run_uses_matched_completion_pairs_and_failure_notes(tmp_path) -> None:
    add_root = tmp_path / "tasks" / "tiny.calc.add" / "repeat_1"
    (add_root / "baseline").mkdir(parents=True)
    (add_root / "tok-universal").mkdir(parents=True)
    (add_root / "baseline" / "run.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "success": False,
                    "notes": ["hidden_tests_failed"],
                    "details": {"execution_contract_met": True},
                },
                "notes": [],
                "provider_usage": {"total_tokens": 100},
            }
        )
    )
    (add_root / "tok-universal" / "run.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "success": False,
                    "notes": ["hidden_tests_failed", "allowed_path_check_failed"],
                    "details": {"execution_contract_met": True},
                },
                "notes": ["read_only_loop_recovery_step_3"],
                "provider_usage": {"total_tokens": 90},
            }
        )
    )

    subtract_root = tmp_path / "tasks" / "tiny.calc.subtract" / "repeat_1"
    (subtract_root / "baseline").mkdir(parents=True)
    (subtract_root / "tok-universal").mkdir(parents=True)
    (subtract_root / "baseline" / "run.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "success": False,
                    "notes": ["hidden_tests_failed"],
                    "details": {"execution_contract_met": False},
                },
                "notes": [],
                "provider_usage": {"total_tokens": 120},
            }
        )
    )
    (subtract_root / "tok-universal" / "run.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "success": False,
                    "notes": ["hidden_tests_failed", "allowed_path_check_failed"],
                    "details": {"execution_contract_met": False},
                },
                "notes": ["read_only_loop_recovery_step_4", "read_only_loop_recovery_step_5"],
                "provider_usage": {"total_tokens": 130},
            }
        )
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
                tool_engagement_stats={
                    "decision_grade": False,
                    "decision_grade_blockers": ["read_only_loop_recovery_asymmetry_matched_pair"],
                    "integrity_asymmetry_flags": ["read_only_loop_recovery_asymmetry"],
                    "integrity_artifact_flags": [],
                },
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
    assert summary["token_cost_decision_grade"] is False
    assert summary["token_cost_decision_gate"]["matched_pair_threshold"] == 15
    assert summary["token_cost_decision_gate"]["passed"] is False
    assert "matched_completion_pairs_below_threshold" in summary["token_cost_decision_reasons"]
    assert summary["fairness_diagnostics"]["execution_contract_compliance_rate"]["baseline"] == 0.5
    assert summary["fairness_diagnostics"]["execution_contract_compliance_rate"]["tok-universal"] == 0.5
    assert summary["fairness_diagnostics"]["loop_recovery_trigger_counts"]["baseline"] == 0
    assert summary["fairness_diagnostics"]["loop_recovery_trigger_counts"]["tok-universal"] == 3
    assert summary["fairness_diagnostics"]["integrity_clean_matched_pair_count"] == 1
    assert summary["fairness_diagnostics"]["integrity_blocked_matched_pair_count"] == 0
    assert summary["fairness_diagnostics"]["matched_pair_blocker_counts"] == {}
    assert summary["fairness_diagnostics"]["matched_pair_blocker_rates"] == {}
    assert summary["fairness_diagnostics"]["matched_pair_blocked_by_asymmetry_count"] == 0
    assert summary["fairness_diagnostics"]["matched_pair_blocked_by_artifact_count"] == 0
    reasons = {item["reason"]: item["count"] for item in summary["top_completion_failure_reasons"]}
    assert reasons["baseline:hidden_tests_failed"] == 2
    assert reasons["tok-universal:hidden_tests_failed"] == 2


def test_summarize_tiny_patch_run_marks_token_cost_decision_grade_with_sufficient_pairs(tmp_path) -> None:
    runs = tuple(
        _run(
            task_id=f"tiny.calc.task{i}",
            repeat_index=1,
            baseline_success=True,
            tok_success=True,
            matched_completion_pair=True,
            total_token_delta=-5,
            tool_engagement_stats={"decision_grade": True},
        )
        for i in range(1, 16)
    )
    catalog_like = SimpleNamespace(runs=runs)

    summary = summarize_tiny_patch_run(catalog_like, output_root=tmp_path)

    assert summary["matched_completion_pair_count"] == 15
    assert summary["token_cost_decision_gate"]["matched_pair_threshold"] == 15
    assert summary["token_cost_decision_gate"]["passed"] is True
    assert summary["token_cost_decision_grade"] is True
    assert summary["token_cost_decision_reasons"] == []


def test_summarize_tiny_patch_run_reports_blocker_breakdown(tmp_path) -> None:
    runs = (
        _run(
            task_id="tiny.calc.blocked.asym",
            repeat_index=1,
            matched_completion_pair=True,
            tool_engagement_stats={
                "decision_grade": False,
                "decision_grade_blockers": ["read_only_loop_recovery_asymmetry_matched_pair"],
                "integrity_asymmetry_flags": ["read_only_loop_recovery_asymmetry"],
                "integrity_artifact_flags": [],
            },
        ),
        _run(
            task_id="tiny.calc.blocked.artifact",
            repeat_index=1,
            matched_completion_pair=True,
            tool_engagement_stats={
                "decision_grade": False,
                "decision_grade_blockers": [],
                "integrity_asymmetry_flags": [],
                "integrity_artifact_flags": ["execution_contract_not_met_artifact"],
            },
        ),
    )
    catalog_like = SimpleNamespace(runs=runs)

    summary = summarize_tiny_patch_run(catalog_like, output_root=tmp_path)

    fairness = summary["fairness_diagnostics"]
    assert fairness["integrity_blocked_matched_pair_count"] == 2
    assert fairness["matched_pair_blocker_counts"]["execution_contract_not_met_artifact"] == 1
    assert fairness["matched_pair_blocker_counts"]["read_only_loop_recovery_asymmetry_matched_pair"] == 1
    assert fairness["matched_pair_blocker_rates"]["execution_contract_not_met_artifact"] == 0.5
    assert fairness["matched_pair_blocker_rates"]["read_only_loop_recovery_asymmetry_matched_pair"] == 0.5
    assert fairness["matched_pair_blocked_by_asymmetry_count"] == 1
    assert fairness["matched_pair_blocked_by_artifact_count"] == 1
    assert fairness["matched_pair_blocked_by_asymmetry_pct"] == 50.0
    assert fairness["matched_pair_blocked_by_artifact_pct"] == 50.0


def test_summarize_tiny_patch_run_emits_controlled_comparator_and_attribution(tmp_path) -> None:
    root = tmp_path / "tasks" / "tiny.calc.add" / "repeat_1"
    (root / "baseline").mkdir(parents=True)
    (root / "tok-universal").mkdir(parents=True)
    (root / "tok-controlled").mkdir(parents=True)

    baseline_payload = {
        "evaluation": {"success": True, "notes": [], "details": {"execution_contract_met": True}},
        "notes": [],
        "tool_calls": 2,
        "provider_usage": {"total_tokens": 100, "prompt_tokens": 70, "completion_tokens": 30},
        "turns": [
            {"provider_usage": {"total_tokens": 100}, "compression_metrics": {}, "response_metrics": {}},
        ],
    }
    tok_universal_payload = {
        "evaluation": {"success": True, "notes": [], "details": {"execution_contract_met": True}},
        "notes": [],
        "tool_calls": 5,
        "provider_usage": {"total_tokens": 150, "prompt_tokens": 110, "completion_tokens": 40},
        "turns": [
            {
                "provider_usage": {"total_tokens": 150},
                "compression_metrics": {
                    "total_saved_tokens": 20,
                    "type_breakdown": {"tool:file_read_overlap_delta": 11},
                    "input_behavior_signals": {
                        "request_policy_tool_compatible": 1,
                        "tok_history_compression_skipped": 1,
                    },
                },
                "response_metrics": {"response_behavior_signals": {}},
                "diagnostics": {
                    "tool_protocol_retry_count": 1,
                    "tool_protocol_retry_success": 1,
                    "tool_protocol_retry_reason": "missing_tool_call_for_call_id",
                },
            },
        ],
    }
    tok_controlled_payload = {
        "evaluation": {"success": True, "notes": [], "details": {"execution_contract_met": True}},
        "notes": [],
        "tool_calls": 3,
        "provider_usage": {"total_tokens": 110, "prompt_tokens": 75, "completion_tokens": 35},
        "turns": [
            {
                "provider_usage": {"total_tokens": 110},
                "compression_metrics": {
                    "total_saved_tokens": 15,
                    "type_breakdown": {"tool:file_reread_diff": 9},
                    "input_behavior_signals": {"tok_history_pairing_safety_degraded": 1},
                },
                "response_metrics": {"response_behavior_signals": {}},
            },
        ],
    }
    (root / "baseline" / "run.json").write_text(json.dumps(baseline_payload))
    (root / "tok-universal" / "run.json").write_text(json.dumps(tok_universal_payload))
    (root / "tok-controlled" / "run.json").write_text(json.dumps(tok_controlled_payload))

    catalog_like = SimpleNamespace(
        runs=(_run(task_id="tiny.calc.add", repeat_index=1, total_token_delta=50, matched_completion_pair=True),)
    )
    summary = summarize_tiny_patch_run(catalog_like, output_root=tmp_path)

    comparators = summary["comparators"]
    assert "tok-universal" in comparators
    assert "tok-controlled" in comparators
    assert comparators["tok-universal"]["token_delta_total_sum"] == 50
    assert comparators["tok-universal"]["compression_recovery_estimate_total"] == 20
    assert comparators["tok-universal"]["behavior_inflation_estimate_total"] == 70
    assert comparators["tok-universal"]["behavior_counters"]["tool_protocol_retry_count"] == 1
    assert comparators["tok-universal"]["behavior_counters"]["tool_protocol_retry_success"] == 1
    assert comparators["tok-universal"]["tool_protocol_retry_reason_counts"]["missing_tool_call_for_call_id"] == 1
    assert comparators["tok-controlled"]["token_delta_total_sum"] == 10
    assert comparators["tok-controlled"]["compression_recovery_estimate_total"] == 15
    assert comparators["tok-controlled"]["behavior_inflation_estimate_total"] == 25
    assert comparators["tok-controlled"]["codec_family_saved_tokens"]["tool:file_reread_diff"] == 9

    attribution = summary["attribution"]
    assert attribution["tok-universal"]["token_delta_total_sum"] == 50
    assert attribution["tok-controlled"]["token_delta_total_sum"] == 10
    stability = summary["stability"]
    assert stability["tok-universal"]["token_delta_total_p50"] == 50.0
    assert stability["tok-universal"]["token_delta_total_p90"] == 50.0
    turn_warnings = summary["turn_diagnostics"]["runtime_contract_warnings"]
    assert turn_warnings["tool_protocol_retry_count"] == 1
    assert turn_warnings["tool_protocol_retry_success"] == 1
    retry_reasons = summary["turn_diagnostics"]["tool_protocol_retry_reasons"]
    assert retry_reasons["missing_tool_call_for_call_id"] == 1
