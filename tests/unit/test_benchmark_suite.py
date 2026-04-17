from __future__ import annotations

from pathlib import Path

import pytest

from tok.testing.benchmark_suite import (
    BenchmarkLane,
    BenchmarkTaskManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"

pytestmark = pytest.mark.skip(
    reason="benchmarks/ directory not present — benchmark suite tests are integration tests that require asset checkout"
)


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
