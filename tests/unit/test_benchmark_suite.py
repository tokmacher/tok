from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tok.cli import app
from tok.testing.benchmark_suite import (
    BENCHMARK_REPORT_STATEMENT,
    BenchmarkComparisonRun,
    BenchmarkLane,
    BenchmarkTaskManifest,
    build_benchmark_report,
    build_condition_plan,
    check_benchmark_report,
    load_benchmark_catalog,
    render_benchmark_report_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"
runner = CliRunner()


def _run(**overrides: object) -> BenchmarkComparisonRun:
    payload: dict[str, object] = {
        "lane_id": "production_claude_lane",
        "task_id": "exec.click.option-precedence",
        "family": "execution_patch",
        "repeat_index": 1,
        "public_release": True,
        "baseline_success": True,
        "tok_success": True,
        "quality_gate_passed": True,
        "total_token_delta": -120,
        "latency_delta_ms": 8.5,
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


def test_load_benchmark_catalog_counts_pilot_assets() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)

    assert catalog.headline_lane().id == "production_claude_lane"
    assert len(catalog.compatibility_lanes()) == 3
    assert catalog.family_counts() == {
        "execution_patch": 6,
        "repo_grounding": 20,
        "real_session": 4,
    }


def test_secondary_lane_requires_adapter_metadata() -> None:
    with pytest.raises(ValueError, match="secondary lanes must declare adapter_name"):
        BenchmarkLane.from_dict(
            {
                "id": "adapter_broken_lane",
                "runtime_path": "UniversalTokRuntime via compatibility shim",
                "transport_shape": "broken",
                "model_family": "broken",
                "provider": "broken",
                "adapter_name": "",
                "adapter_notes": "",
                "claim_scope": "secondary",
                "normalized_differences": [],
            }
        )


def test_prompt_leak_validation_rejects_forbidden_terms() -> None:
    with pytest.raises(ValueError, match="prompt leaks forbidden terms"):
        BenchmarkTaskManifest.from_dict(
            {
                "id": "exec.leaky.task",
                "family": "execution_patch",
                "title": "Leaky prompt",
                "summary": "Bad task",
                "repo": "tok",
                "ref": "HEAD",
                "setup_script": "no_setup_required",
                "prompt": "Please edit src/tok/runtime/pipeline/request_validation.py directly.",
                "allowed_tools": ["view_file", "edit_file"],
                "time_budget_minutes": 15,
                "step_budget": 60,
                "success_evaluator": {"kind": "execution_patch"},
                "artifact_policy": {"publish_diff": True},
                "public_release": False,
                "asset_dir": "assets/exec.leaky.task",
                "workspace_source": {"kind": "asset_snapshot", "path": "assets/exec.leaky.task/workspace"},
                "family_payload": {
                    "allowed_paths": ["src/tok/runtime/pipeline/request_validation.py"],
                    "visible_tests": [],
                    "seed_patch_path": "seed.patch",
                },
                "seed_patch": "seed.patch",
                "prompt_forbidden_terms": ["src/tok/runtime/pipeline/request_validation.py"],
                "allowed_paths": ["src/tok/runtime/pipeline/request_validation.py"],
                "hidden_tests": ["tests/unit/test_request_validation.py::test_name"],
            }
        )


def test_build_condition_plan_only_toggles_runtime_wrapper() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)

    baseline = build_condition_plan(catalog, lane_id="production_claude_lane", condition="baseline")
    tok = build_condition_plan(catalog, lane_id="production_claude_lane", condition="tok-universal")

    assert baseline.task_ids == tok.task_ids
    assert baseline.runtime_path == tok.runtime_path
    assert baseline.transport_shape == tok.transport_shape
    assert baseline.runtime_wrapper_active is False
    assert tok.runtime_wrapper_active is True
    assert baseline.candidate_mode == "baseline"
    assert tok.candidate_mode == "tok-universal"


def test_build_benchmark_report_separates_public_and_supplemental_sections() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    runs = [
        _run(repeat_index=1),
        _run(repeat_index=2, total_token_delta=-140, latency_delta_ms=7.0),
        _run(
            lane_id="adapter_openai_chat_lane",
            task_id="qa.click.option-precedence",
            family="repo_grounding",
            repeat_index=1,
            total_token_delta=-40,
            latency_delta_ms=3.0,
        ),
    ]

    report = build_benchmark_report(catalog, runs, title="Pilot Production Report")
    markdown = render_benchmark_report_markdown(report)

    assert BENCHMARK_REPORT_STATEMENT in markdown
    assert "## Public Production Lane" in markdown
    assert "## Supplemental Internal/Advisory Tasks" in markdown
    assert "production_claude_lane" in markdown
    assert "adapter_openai_chat_lane" in markdown
    assert "tok-native" not in markdown
    assert "tok-neuro" not in markdown


def test_headline_public_claim_is_disabled_on_regression_or_instability(tmp_path: Path) -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    report = build_benchmark_report(
        catalog,
        [
            _run(tok_success=False, quality_gate_passed=False, paired_result_stable=False),
        ],
    )

    headline = report.headline_summary()
    assert headline.consistency_gate_passed is False
    assert headline.public_claim_allowed is False
    assert "success_regressed_vs_baseline" in headline.notes
    assert "paired_result_unstable" in headline.notes

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    checked = check_benchmark_report(report_path)
    assert checked["passed"] is False
    assert checked["headline_lane"] == "production_claude_lane"


def test_headline_public_claim_is_disabled_when_all_runs_fail_even_without_delta_regression() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    report = build_benchmark_report(
        catalog,
        [
            _run(
                baseline_success=False, tok_success=False, baseline_grounding_success=False, tok_grounding_success=False
            ),
        ],
    )

    headline = report.headline_summary()
    assert headline.consistency_gate_passed is False
    assert headline.public_claim_allowed is False
    assert "tok_success_below_absolute_floor" in headline.notes
    assert "tok_grounding_below_absolute_floor" in headline.notes
    assert "all_conditions_failed" in headline.notes


def test_headline_public_claim_requires_quality_gate_floor() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    report = build_benchmark_report(
        catalog,
        [
            _run(quality_gate_passed=False),
        ],
    )

    headline = report.headline_summary()
    assert headline.consistency_gate_passed is False
    assert headline.public_claim_allowed is False
    assert "quality_gate_below_absolute_floor" in headline.notes


def test_summary_keeps_advisory_format_violations_separate_from_completion_failures() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    report = build_benchmark_report(
        catalog,
        [
            _run(
                quality_gate_passed=True,
                baseline_success=True,
                tok_success=True,
                format_contract_violations=["evidence_block_count", "invalid_citations"],
            ),
        ],
    )

    headline = report.headline_summary()
    assert headline.consistency_gate_passed is True
    assert headline.public_claim_allowed is True
    assert headline.format_contract_violations["evidence_block_count"] == 1
    assert headline.format_contract_violations["invalid_citations"] == 1
    assert "advisory_format_contract_violations_present" in headline.notes


def test_repo_grounding_manifest_allows_zero_min_grounded_retrieval_steps() -> None:
    manifest = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.relaxed-grounding",
            "family": "repo_grounding",
            "title": "Relaxed grounding",
            "summary": "Completion-first manifest validation.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined?",
            "allowed_tools": ["view_file", "grep_search"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "repo_grounding", "min_grounded_retrieval_steps": 0},
            "artifact_policy": {"publish_answer": True},
            "public_release": False,
            "asset_dir": "assets/qa.local.relaxed-grounding",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["src/app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [{"file": "src/app.py", "anchor": "def answer_symbol", "why": "impl"}],
            "answer_contract": "Answer with concise grounding.",
        }
    )
    assert manifest.success_evaluator["min_grounded_retrieval_steps"] == 0


def test_internal_headline_runs_stay_out_of_public_production_summary() -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    report = build_benchmark_report(
        catalog,
        [
            _run(task_id="qa.click.option-precedence", family="repo_grounding", public_release=True),
            _run(
                task_id="exec.tok.bridge-canonicalization",
                family="execution_patch",
                public_release=False,
                tok_success=False,
                quality_gate_passed=False,
                paired_result_stable=False,
            ),
        ],
    )

    headline = report.headline_summary()
    supplemental = report.supplemental_summaries()

    assert headline.sample_size == 1
    assert headline.public_claim_allowed is True
    assert any(summary.lane.id == "production_claude_lane" for summary in supplemental)
    assert all(summary.public_claim_allowed is False for summary in supplemental)


def test_benchmark_validate_cli_reports_catalog_counts() -> None:
    result = runner.invoke(app, ["dev", "benchmark-validate", "--root", str(BENCHMARK_ROOT)])

    assert result.exit_code == 0
    assert "Benchmark catalog valid" in result.output
    assert "execution_patch" in result.output
    assert "repo_grounding" in result.output
    assert "real_session" in result.output


def test_benchmark_report_cli_renders_from_raw_runs(tmp_path: Path) -> None:
    payload = {
        "title": "Pilot Production Report",
        "runs": [
            _run(repeat_index=1).to_dict(),
            _run(repeat_index=2, total_token_delta=-90, latency_delta_ms=5.0).to_dict(),
            _run(
                lane_id="adapter_gemini_compat_lane",
                task_id="qa.rich.export-path",
                family="repo_grounding",
                repeat_index=1,
                total_token_delta=-15,
                latency_delta_ms=2.0,
            ).to_dict(),
        ],
    }
    input_path = tmp_path / "raw_runs.json"
    output_path = tmp_path / "report.md"
    input_path.write_text(json.dumps(payload, indent=2))

    result = runner.invoke(
        app,
        [
            "dev",
            "benchmark-report",
            "--root",
            str(BENCHMARK_ROOT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    rendered = output_path.read_text()
    assert BENCHMARK_REPORT_STATEMENT in rendered
    assert "Supplemental Internal/Advisory Tasks" in rendered
    assert "adapter_gemini_compat_lane" in rendered


def test_benchmarks_tree_has_no_unapproved_evaluators_directory() -> None:
    assert not (BENCHMARK_ROOT / "evaluators").exists()
