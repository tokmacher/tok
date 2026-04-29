from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tok.cli import app
from tok.testing.benchmark_executor import (
    BenchmarkTaskRunResult,
    BenchmarkToolExecutor,
    CatalogBenchmarkRun,
    FamilyEvaluator,
    MaterializedBenchmarkTask,
    TaskEvaluationResult,
    TaskMaterializer,
    ToolExecutionRecord,
    ToolLoopExecutor,
    _directory_sha256,
    _execution_failure_code,
    _extract_text_tool_calls,
    _normalize_pytest_command,
    run_catalog_benchmark_suite,
)
from tok.testing.benchmark_suite import (
    BenchmarkCatalog,
    BenchmarkComparisonRun,
    BenchmarkLane,
    BenchmarkTaskManifest,
    build_benchmark_report,
)
from tok.testing.live_benchmark import LiveBenchmarkRunner, ProviderUsageSnapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"
runner = CliRunner()


class _SequencedCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def create(self, **kwargs):
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            ),
        )


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_SequencedCompletions(responses))


class _StepRunner:
    model: str = "test-model"

    def __init__(self, steps: list[dict[str, object]]) -> None:
        self._steps = list(steps)

    def run_conversation_step(self, **kwargs):
        del kwargs
        if self._steps:
            payload = self._steps.pop(0)
        else:
            # CI can exercise an extra recovery loop iteration on some Python
            # versions. Fall back to a terminal assistant turn instead of
            # crashing the test harness with IndexError.
            payload = {"raw_response": "Done.", "visible_response": "Done.", "content_blocks": []}
        return SimpleNamespace(
            provider_usage=ProviderUsageSnapshot(
                prompt_tokens=int(payload.get("prompt_tokens", 10)),
                completion_tokens=int(payload.get("completion_tokens", 10)),
                total_tokens=int(payload.get("total_tokens", 20)),
                latency_ms=float(payload.get("latency_ms", 1.0)),
            ),
            response_metrics={"response_behavior_signals": {}},
            compression_metrics={},
            content_blocks=list(payload.get("content_blocks", [])),
            visible_response=str(payload.get("visible_response", "")),
            raw_response=str(payload.get("raw_response", "")),
        )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _production_lane() -> BenchmarkLane:
    return BenchmarkLane.from_dict(
        {
            "id": "production_claude_lane",
            "runtime_path": "Bridge-first Claude Code flow through UniversalTokRuntime using the production default request/response path.",
            "transport_shape": "claude_code_bridge_messages",
            "model_family": "claude",
            "provider": "anthropic",
            "adapter_name": "",
            "adapter_notes": "",
            "claim_scope": "headline",
            "normalized_differences": [],
        }
    )


def _task_run_result(
    *,
    condition: str,
    success: bool,
    grounding_success: bool,
    tool_calls: int,
    family: str = "repo_grounding",
    eval_notes: tuple[str, ...] = (),
    eval_details: dict[str, object] | None = None,
    local_failure: str = "",
    run_notes: tuple[str, ...] = (),
) -> BenchmarkTaskRunResult:
    return BenchmarkTaskRunResult(
        lane_id="production_claude_lane",
        condition=condition,
        task_id="qa.local.answer",
        family=family,
        repeat_index=1,
        workspace_root="/tmp/workspace",
        answer_text="answer",
        raw_response="answer",
        provider_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "latency_ms": 1.0},
        tool_calls=tool_calls,
        invalid_tool_calls=0,
        reacquisition_events=0,
        clean_exit=True,
        modified_files=tuple(),
        tool_records=tuple(),
        turns=tuple(),
        evaluation=TaskEvaluationResult(
            success=success,
            grounding_success=grounding_success,
            details=eval_details or {},
            notes=eval_notes,
        ),
        local_failure=local_failure,
        notes=run_notes,
    )


def _execution_patch_task(*, workspace: Path, step_budget: int = 6) -> BenchmarkTaskManifest:
    return BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["list_dir", "view_file", "grep_search", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": step_budget,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )


def test_extract_text_tool_calls_supports_fenced_json_variants() -> None:
    raw = """
<tool_name>view_file</tool_name>
<parameter name="path">src/app.py</parameter>
@Tool grep_search {pattern: answer_symbol, path: src}
Tool use (list_dir): {path: tests}
```json
{"tool":"run_tests","args":{"command":"python -m pytest -q tests/test_app.py::test_answer_symbol"}}
```
```json
{"name":"git_diff","path":"src/app.py"}
```
"""
    blocks = _extract_text_tool_calls(
        raw,
        ("view_file", "grep_search", "list_dir", "run_tests", "git_diff"),
    )
    names = [str(block["name"]) for block in blocks]
    assert "view_file" in names
    assert "grep_search" in names
    assert "list_dir" in names
    assert "run_tests" in names
    assert "git_diff" in names
    run_tests_block = next(block for block in blocks if block["name"] == "run_tests")
    assert run_tests_block["input"]["command"].startswith("python -m pytest")


def test_tool_loop_executor_extracts_text_tools_in_tok_universal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer_symbol():\n    return 'ok'\n")
    (workspace / "tests" / "test_app.py").write_text("def test_answer_symbol():\n    assert True\n")
    (workspace / "gold_answer.json").write_text(
        json.dumps(
            {
                "required_files": ["src/app.py", "tests/test_app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [
                    {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                    {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
                ],
            }
        )
    )

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.answer",
            "family": "repo_grounding",
            "title": "Grounded local answer",
            "summary": "Answer with evidence.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined?",
            "allowed_tools": ["list_dir", "view_file", "grep_search"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "repo_grounding", "min_grounded_retrieval_steps": 2},
            "artifact_policy": {"publish_answer": True},
            "public_release": False,
            "asset_dir": str(tmp_path / "assets"),
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {
                "gold_answer_path": "gold_answer.json",
                "required_files": ["src/app.py", "tests/test_app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [
                    {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                    {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
                ],
            },
            "required_files": ["src/app.py", "tests/test_app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [
                {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="tok-universal",
        asset_root=str(tmp_path / "assets"),
        workspace_root=str(workspace),
        resolved_ref="deadbeef",
        reportable=False,
        setup_ran=False,
    )
    runner = _StepRunner(
        [
            {
                "raw_response": '@Tool list_dir {path: "."}\n@Tool view_file {path: "src/app.py", start: 1, end: 5}',
                "visible_response": "",
                "content_blocks": [],
            },
            {
                "raw_response": (
                    "answer_symbol is defined in src/app.py and validated in tests/test_app.py.\n"
                    "Evidence:\n"
                    "- src/app.py line 1: def answer_symbol defines the helper.\n"
                    "- tests/test_app.py line 1: test_answer_symbol validates the behavior."
                ),
                "visible_response": (
                    "answer_symbol is defined in src/app.py and validated in tests/test_app.py.\n"
                    "Evidence:\n"
                    "- src/app.py line 1: def answer_symbol defines the helper.\n"
                    "- tests/test_app.py line 1: test_answer_symbol validates the behavior."
                ),
                "content_blocks": [],
            },
        ]
    )
    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.tool_calls > 0
    assert any(note == "text_tool_extraction_step_1" for note in result.notes)
    assert result.clean_exit is True


def test_tool_loop_executor_allows_premature_final_for_repo_grounding(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer_symbol():\n    return 'ok'\n")
    (workspace / "tests" / "test_app.py").write_text("def test_answer_symbol():\n    assert True\n")
    (workspace / "gold_answer.json").write_text(
        json.dumps(
            {
                "required_files": ["src/app.py", "tests/test_app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [
                    {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                    {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
                ],
            }
        )
    )

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.answer",
            "family": "repo_grounding",
            "title": "Grounded local answer",
            "summary": "Answer with evidence.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined?",
            "allowed_tools": ["list_dir", "view_file", "grep_search"],
            "time_budget_minutes": 1,
            "step_budget": 5,
            "success_evaluator": {"kind": "repo_grounding", "min_grounded_retrieval_steps": 2},
            "artifact_policy": {"publish_answer": True},
            "public_release": False,
            "asset_dir": str(tmp_path / "assets"),
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {
                "gold_answer_path": "gold_answer.json",
                "required_files": ["src/app.py", "tests/test_app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [
                    {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                    {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
                ],
            },
            "required_files": ["src/app.py", "tests/test_app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [
                {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path / "assets"),
        workspace_root=str(workspace),
        resolved_ref="deadbeef",
        reportable=False,
        setup_ran=False,
    )
    runner = _StepRunner(
        [
            {
                "raw_response": "I can answer this quickly without tools.",
                "visible_response": "I can answer this quickly without tools.",
                "content_blocks": [],
            },
            {
                "raw_response": '@Tool list_dir {path: "."}\n@Tool view_file {path: "src/app.py", start: 1, end: 5}',
                "visible_response": "",
                "content_blocks": [],
            },
            {
                "raw_response": (
                    "answer_symbol is defined in src/app.py and validated in tests/test_app.py.\n"
                    "Evidence:\n"
                    "- src/app.py line 1: def answer_symbol defines the helper.\n"
                    "- tests/test_app.py line 1: test_answer_symbol validates the behavior."
                ),
                "visible_response": (
                    "answer_symbol is defined in src/app.py and validated in tests/test_app.py.\n"
                    "Evidence:\n"
                    "- src/app.py line 1: def answer_symbol defines the helper.\n"
                    "- tests/test_app.py line 1: test_answer_symbol validates the behavior."
                ),
                "content_blocks": [],
            },
        ]
    )
    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert "premature_final_step_1" not in result.notes
    assert len(result.turns) == 1
    assert result.clean_exit is True
    assert result.tool_calls == 0


def test_tool_loop_executor_execution_patch_requires_edit_and_run_tests_before_finalizing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = _execution_patch_task(workspace=workspace, step_budget=8)
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    runner = _StepRunner(
        [
            {
                "raw_response": "I am done.",
                "visible_response": "I am done.",
                "content_blocks": [],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step2_edit",
                        "name": "edit_file",
                        "input": {
                            "path": "src/app.py",
                            "old_string": "return 1",
                            "new_string": "return 2",
                        },
                    }
                ],
            },
            {
                "raw_response": "Patch applied.",
                "visible_response": "Patch applied.",
                "content_blocks": [],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step4_tests",
                        "name": "run_tests",
                        "input": {"command": "python -m pytest -q tests/test_app.py::test_answer"},
                    }
                ],
            },
            {
                "raw_response": "Done.",
                "visible_response": "Done.",
                "content_blocks": [],
            },
            {
                # Keep one spare terminal turn so execution-patch recovery
                # nudges cannot exhaust the scripted runner on slower/variant
                # CI paths (for example Python-version-specific step timing).
                "raw_response": "Done.",
                "visible_response": "Done.",
                "content_blocks": [],
            },
        ]
    )

    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.success is True
    assert "premature_final_step_1" in result.notes
    assert "premature_final_step_3" in result.notes
    assert "execution_contract_met" in result.notes
    assert result.clean_exit is True


def test_tool_loop_executor_execution_patch_triggers_read_only_loop_recovery(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = _execution_patch_task(workspace=workspace, step_budget=10)
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="tok-universal",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    list_dir_call = {
        "type": "tool_use",
        "id": "list_dir_read",
        "name": "list_dir",
        "input": {"path": "."},
    }
    runner = _StepRunner(
        [
            {"raw_response": "", "visible_response": "", "content_blocks": [list_dir_call]},
            {"raw_response": "", "visible_response": "", "content_blocks": [list_dir_call]},
            {"raw_response": "", "visible_response": "", "content_blocks": [list_dir_call]},
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step4_edit",
                        "name": "edit_file",
                        "input": {
                            "path": "src/app.py",
                            "old_string": "return 1",
                            "new_string": "return 2",
                        },
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step5_tests",
                        "name": "run_tests",
                        "input": {"command": "python -m pytest -q tests/test_app.py::test_answer"},
                    }
                ],
            },
            {"raw_response": "Done.", "visible_response": "Done.", "content_blocks": []},
        ]
    )

    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.success is True
    assert "read_only_loop_recovery_step_3" in result.notes
    assert "read_only_loop_recovery_count_1" in result.notes
    assert "execution_contract_met" in result.notes


def test_tool_loop_executor_execution_patch_blocks_finalize_after_failed_tests(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = _execution_patch_task(workspace=workspace, step_budget=8)
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="tok-universal",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    runner = _StepRunner(
        [
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step1_edit_bad_indent",
                        "name": "edit_file",
                        "input": {
                            "path": "src/app.py",
                            "old_string": "return 1",
                            "new_string": "return * 2",
                        },
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step2_tests_fail",
                        "name": "run_tests",
                        "input": {"command": "python -m pytest -q tests/test_app.py::test_answer"},
                    }
                ],
            },
            {
                "raw_response": "All good now.",
                "visible_response": "All good now.",
                "content_blocks": [],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step4_fix_indent",
                        "name": "edit_file",
                        "input": {
                            "path": "src/app.py",
                            "old_string": "return * 2",
                            "new_string": "return 2",
                        },
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step5_tests_pass",
                        "name": "run_tests",
                        "input": {"command": "python -m pytest -q tests/test_app.py::test_answer"},
                    }
                ],
            },
            {
                "raw_response": "Done.",
                "visible_response": "Done.",
                "content_blocks": [],
            },
        ]
    )

    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.success is True
    assert "premature_final_step_3" in result.notes
    assert "premature_final_after_failed_tests_step_3" in result.notes
    assert result.clean_exit is True


def test_tool_loop_executor_execution_patch_recovers_after_edit_search_miss(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = _execution_patch_task(workspace=workspace, step_budget=8)
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    runner = _StepRunner(
        [
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step1_bad_edit",
                        "name": "edit_file",
                        "input": {
                            "path": "src/app.py",
                            "old_string": "return 999",
                            "new_string": "return 2",
                        },
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step2_good_edit",
                        "name": "edit_file",
                        "input": {
                            "path": "src/app.py",
                            "old_string": "return 1",
                            "new_string": "return 2",
                        },
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step3_tests",
                        "name": "run_tests",
                        "input": {"command": "python -m pytest -q tests/test_app.py::test_answer"},
                    }
                ],
            },
            {
                "raw_response": "Done.",
                "visible_response": "Done.",
                "content_blocks": [],
            },
        ]
    )

    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.success is True
    assert "edit_file_search_miss_recovery_step_1" in result.notes
    assert "edit_file_search_miss_recovery_count_1" in result.notes
    assert "execution_contract_met" in result.notes


def test_tool_loop_executor_execution_patch_escalates_edit_search_miss_recovery(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = _execution_patch_task(workspace=workspace, step_budget=10)
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    runner = _StepRunner(
        [
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step1_bad_edit",
                        "name": "edit_file",
                        "input": {"path": "src/app.py", "old_string": "return 11", "new_string": "return 2"},
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step2_bad_edit",
                        "name": "edit_file",
                        "input": {"path": "src/app.py", "old_string": "return 12", "new_string": "return 2"},
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step3_bad_edit",
                        "name": "edit_file",
                        "input": {"path": "src/app.py", "old_string": "return 13", "new_string": "return 2"},
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step4_good_edit",
                        "name": "edit_file",
                        "input": {"path": "src/app.py", "old_string": "return 1", "new_string": "return 2"},
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step5_tests",
                        "name": "run_tests",
                        "input": {"command": "python -m pytest -q tests/test_app.py::test_answer"},
                    }
                ],
            },
            {"raw_response": "Done.", "visible_response": "Done.", "content_blocks": []},
        ]
    )

    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.success is True
    assert "edit_file_search_miss_recovery_step_1" in result.notes
    assert "edit_file_search_miss_recovery_step_2" in result.notes
    assert "edit_file_search_miss_recovery_step_3" in result.notes
    assert "edit_file_search_miss_recovery_count_3" in result.notes


def test_tool_loop_executor_execution_patch_suppresses_redundant_run_tests_without_new_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = _execution_patch_task(workspace=workspace, step_budget=10)
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="tok-universal",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    test_cmd = "python -m pytest -q tests/test_app.py::test_answer"
    runner = _StepRunner(
        [
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "step1_edit",
                        "name": "edit_file",
                        "input": {"path": "src/app.py", "old_string": "return 1", "new_string": "return 2"},
                    }
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {"type": "tool_use", "id": "step2_tests", "name": "run_tests", "input": {"command": test_cmd}}
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {"type": "tool_use", "id": "step3_tests", "name": "run_tests", "input": {"command": test_cmd}}
                ],
            },
            {
                "raw_response": "",
                "visible_response": "",
                "content_blocks": [
                    {"type": "tool_use", "id": "step4_tests", "name": "run_tests", "input": {"command": test_cmd}}
                ],
            },
            {"raw_response": "Done.", "visible_response": "Done.", "content_blocks": []},
        ]
    )

    result = ToolLoopExecutor(runner, catalog_root=tmp_path).run_task(materialized, output_root=tmp_path / "out")

    assert result.success is True
    assert "redundant_run_tests_suppressed_step_4" in result.notes
    assert "redundant_run_tests_suppressed_count_1" in result.notes
    assert any(
        "run_tests suppressed: this command already succeeded without a new edit." in record.content_preview
        for record in result.tool_records
    )


def test_compare_pair_uses_completion_comparability_and_tracks_advisories(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.answer",
            "family": "repo_grounding",
            "title": "Grounded local answer",
            "summary": "Answer with evidence.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined?",
            "allowed_tools": ["grep_search", "view_file"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "repo_grounding",
                "min_grounded_retrieval_steps": 2,
                "requires_evidence_block": True,
            },
            "artifact_policy": {"publish_answer": True},
            "public_release": True,
            "asset_dir": "assets/qa.local.answer",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/qa.local.answer/workspace"},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["src/app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [{"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"}],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )

    comparison_no_tools = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(condition="baseline", success=False, grounding_success=False, tool_calls=0),
        candidate=_task_run_result(
            condition="tok-universal",
            success=True,
            grounding_success=True,
            tool_calls=0,
            eval_notes=("evidence_block_count", "invalid_citations"),
        ),
    )
    assert comparison_no_tools.quality_gate_passed is True
    assert comparison_no_tools.format_contract_violations == ("evidence_block_count", "invalid_citations")
    assert comparison_no_tools.tool_engagement_stats["tok_tool_calls"] == 0

    comparison_candidate_quality = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(condition="baseline", success=False, grounding_success=False, tool_calls=2),
        candidate=_task_run_result(condition="tok-universal", success=True, grounding_success=True, tool_calls=2),
    )
    assert comparison_candidate_quality.quality_gate_passed is True

    comparison_bad_candidate = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(condition="baseline", success=False, grounding_success=False, tool_calls=2),
        candidate=_task_run_result(condition="tok-universal", success=False, grounding_success=False, tool_calls=2),
    )
    assert comparison_bad_candidate.quality_gate_passed is False


def test_repo_grounding_completion_passes_without_evidence_block_when_core_facts_present(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.answer",
            "family": "repo_grounding",
            "title": "Grounded local answer",
            "summary": "Answer with evidence.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined?",
            "allowed_tools": ["grep_search", "view_file"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "repo_grounding", "min_grounded_retrieval_steps": 2},
            "artifact_policy": {"publish_answer": True},
            "public_release": False,
            "asset_dir": str(tmp_path),
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["src/app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [{"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"}],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    (tmp_path / "gold_answer.json").write_text(
        json.dumps(
            {
                "required_files": ["src/app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [{"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"}],
            }
        )
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(tmp_path),
        resolved_ref="deadbeef",
        reportable=False,
        setup_ran=False,
    )
    result = evaluator.evaluate(
        materialized,
        answer_text="answer_symbol is defined in src/app.py.",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=0,
        tool_records=[],
        workspace_root=tmp_path,
    )
    assert result.success is True
    assert "evidence_block_count" in result.notes
    assert "invalid_citations" in result.notes


def test_execution_patch_requires_clean_exit_for_success(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 2\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": True},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    result = evaluator.evaluate(
        materialized,
        answer_text="done",
        clean_exit=False,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[
            ToolExecutionRecord(
                step_index=1,
                tool_name="edit_file",
                canonical_tool_name="edit_file",
                tool_input={"path": "src/app.py"},
                invalid=False,
                is_error=False,
                content_preview="Edited src/app.py",
            ),
            ToolExecutionRecord(
                step_index=2,
                tool_name="run_tests",
                canonical_tool_name="run_tests",
                tool_input={"command": "python -m pytest -q tests/test_app.py::test_answer"},
                invalid=False,
                is_error=False,
                content_preview="$ python -m pytest -q tests/test_app.py::test_answer",
            ),
        ],
        workspace_root=workspace,
    )
    assert result.success is False
    assert "clean_exit_required_for_success" in result.notes


def test_run_tests_reports_failure_exit_and_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer() -> int:\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer() -> None:\n    assert answer() == 2\n"
    )
    executor = BenchmarkToolExecutor(workspace, allowed_tools=("run_tests",), timeout_seconds=30)

    content, is_error, signal = executor._run_tests({"command": "python -m pytest -q tests/test_app.py::test_answer"})

    assert is_error is True
    assert signal == "command_failed"
    assert "$ python -m pytest -q tests/test_app.py::test_answer" in content
    assert "FAILED" in content or "AssertionError" in content


def test_normalize_pytest_command_accepts_supported_prefixes() -> None:
    assert _normalize_pytest_command("pytest -q tests/test_app.py::test_answer") == (
        "python -m pytest -q tests/test_app.py::test_answer"
    )
    assert _normalize_pytest_command("python -m pytest -q tests/test_app.py::test_answer") == (
        "python -m pytest -q tests/test_app.py::test_answer"
    )
    assert _normalize_pytest_command("uv run pytest -q tests/test_app.py::test_answer") == (
        "python -m pytest -q tests/test_app.py::test_answer"
    )
    assert _normalize_pytest_command("uv run python -m pytest -q tests/test_app.py::test_answer") == (
        "python -m pytest -q tests/test_app.py::test_answer"
    )
    assert _normalize_pytest_command("cd ./workspace && pytest -q tests/test_app.py::test_answer") == (
        "python -m pytest -q tests/test_app.py::test_answer"
    )
    assert _normalize_pytest_command("cd /tmp/workspace && python -m pytest -q tests/test_app.py::test_answer") == (
        "python -m pytest -q tests/test_app.py::test_answer"
    )


def test_normalize_pytest_command_rejects_non_pytest_commands() -> None:
    assert _normalize_pytest_command("python -m unittest -q") is None
    assert _normalize_pytest_command("bash -lc 'echo hi'") is None


def test_run_tests_rejects_non_pytest_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    executor = BenchmarkToolExecutor(workspace, allowed_tools=("run_tests",), timeout_seconds=30)

    content, is_error, signal = executor._run_tests({"command": "python -m unittest -q"})

    assert is_error is True
    assert signal == "invalid_tool"
    assert "only pytest commands are allowed" in content


def test_run_tests_accepts_cd_wrapped_pytest_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer() -> int:\n    return 2\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer() -> None:\n    assert answer() == 2\n"
    )
    executor = BenchmarkToolExecutor(workspace, allowed_tools=("run_tests",), timeout_seconds=30)

    content, is_error, signal = executor._run_tests(
        {"command": f"cd {workspace} && python -m pytest -q tests/test_app.py::test_answer"}
    )

    assert is_error is False
    assert signal == ""
    assert "$ python -m pytest -q tests/test_app.py::test_answer" in content


def test_modified_files_parsing_preserves_leading_path_chars(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    def _fake_git(args: list[str], cwd: Path):
        del args, cwd
        return subprocess.CompletedProcess(
            "git status",
            0,
            " M src/calculator.py\n M tests/test_calculator.py\nR  src/old_name.py -> src/new_name.py\n",
            "",
        )

    monkeypatch.setattr("tok.testing.benchmark_executor._tool_executor._git", _fake_git)
    executor = BenchmarkToolExecutor(workspace, allowed_tools=("view_file",), timeout_seconds=30)

    assert executor.modified_files() == (
        "src/calculator.py",
        "src/new_name.py",
        "tests/test_calculator.py",
    )


def test_family_evaluator_modified_files_parsing_preserves_leading_chars(monkeypatch, tmp_path: Path) -> None:
    def _fake_git(args: list[str], cwd: Path):
        del args, cwd
        return subprocess.CompletedProcess(
            "git status",
            0,
            " M src/calculator.py\n M tests/test_calculator.py\n",
            "",
        )

    monkeypatch.setattr("tok.testing.benchmark_executor._evaluator._git", _fake_git)
    evaluator = FamilyEvaluator(catalog_root=tmp_path)

    assert evaluator._modified_files(tmp_path) == ("src/calculator.py", "tests/test_calculator.py")


def test_execution_patch_evaluator_enforces_expect_initial_hidden_failure_gate(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 2\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "execution_patch",
                "clean_exit_required": False,
                "expect_initial_hidden_failure": True,
            },
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )

    result = evaluator.evaluate(
        materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=1,
        tool_records=[
            ToolExecutionRecord(
                step_index=1,
                tool_name="edit_file",
                canonical_tool_name="edit_file",
                tool_input={"path": "src/app.py"},
                invalid=False,
                is_error=False,
                content_preview="Edited src/app.py",
            ),
            ToolExecutionRecord(
                step_index=2,
                tool_name="run_tests",
                canonical_tool_name="run_tests",
                tool_input={"command": "python -m pytest -q tests/test_app.py::test_answer"},
                invalid=False,
                is_error=False,
                content_preview="$ python -m pytest -q tests/test_app.py::test_answer",
            ),
        ],
        workspace_root=workspace,
    )

    assert result.success is False
    assert "initial_hidden_failure_not_observed" in result.notes
    assert result.details["command_invoked"] is True
    assert result.details["command_argv"][:2] == ["/bin/zsh", "-lc"]
    assert isinstance(result.details["command_returncode"], int)
    assert result.details["suspicious_noop_pass"] is True


def test_execution_patch_evaluator_reports_hidden_failure_then_passes_after_fix(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )

    failing = evaluator.evaluate(
        materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=1,
        tool_records=[
            ToolExecutionRecord(
                step_index=1,
                tool_name="edit_file",
                canonical_tool_name="edit_file",
                tool_input={"path": "src/app.py"},
                invalid=False,
                is_error=False,
                content_preview="Edited src/app.py",
            ),
            ToolExecutionRecord(
                step_index=2,
                tool_name="run_tests",
                canonical_tool_name="run_tests",
                tool_input={"command": "python -m pytest -q tests/test_app.py::test_answer"},
                invalid=False,
                is_error=False,
                content_preview="$ python -m pytest -q tests/test_app.py::test_answer",
            ),
        ],
        workspace_root=workspace,
    )
    assert failing.success is False
    assert "hidden_tests_failed" in failing.notes

    (workspace / "src" / "app.py").write_text("def answer():\n    return 2\n")
    passing = evaluator.evaluate(
        materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=1,
        tool_records=[
            ToolExecutionRecord(
                step_index=1,
                tool_name="edit_file",
                canonical_tool_name="edit_file",
                tool_input={"path": "src/app.py"},
                invalid=False,
                is_error=False,
                content_preview="Edited src/app.py",
            ),
            ToolExecutionRecord(
                step_index=2,
                tool_name="run_tests",
                canonical_tool_name="run_tests",
                tool_input={"command": "python -m pytest -q tests/test_app.py::test_answer"},
                invalid=False,
                is_error=False,
                content_preview="$ python -m pytest -q tests/test_app.py::test_answer",
            ),
        ],
        workspace_root=workspace,
    )
    assert passing.success is True
    assert "hidden_tests_failed" not in passing.notes


def test_execution_patch_evaluator_requires_execution_contract_for_success(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 2\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": str(workspace)},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=_production_lane(),
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )

    result = evaluator.evaluate(
        materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=0,
        tool_records=[],
        workspace_root=workspace,
    )

    assert result.success is False
    assert "execution_contract_not_met" in result.notes


def test_compare_pair_marks_integrity_artifacts_as_non_decision_grade(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )

    comparison = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(
            condition="baseline",
            success=True,
            grounding_success=True,
            tool_calls=2,
            eval_details={"command_invoked": True},
        ),
        candidate=_task_run_result(
            condition="tok-universal",
            success=True,
            grounding_success=True,
            tool_calls=2,
            eval_details={"command_invoked": False},
            local_failure="adapter_payload_contract_error",
        ),
    )

    assert comparison.quality_gate_passed is False
    assert "command_not_executed_artifact" in comparison.tool_engagement_stats["integrity_artifact_flags"]
    assert "adapter_payload_contract_error_artifact" in comparison.tool_engagement_stats["integrity_artifact_flags"]
    assert "command_not_executed_asymmetry" in comparison.tool_engagement_stats["integrity_asymmetry_flags"]
    assert "adapter_payload_contract_error_asymmetry" in comparison.tool_engagement_stats["integrity_asymmetry_flags"]


def test_compare_pair_tracks_execution_contract_and_loop_recovery_asymmetry(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )

    comparison = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(
            condition="baseline",
            family="execution_patch",
            success=True,
            grounding_success=True,
            tool_calls=3,
            eval_details={"command_invoked": True, "execution_contract_met": True},
            run_notes=("read_only_loop_recovery_step_3",),
        ),
        candidate=_task_run_result(
            condition="tok-universal",
            family="execution_patch",
            success=True,
            grounding_success=True,
            tool_calls=3,
            eval_details={"command_invoked": True, "execution_contract_met": False},
        ),
    )

    assert comparison.quality_gate_passed is False
    assert "execution_contract_not_met_artifact" in comparison.tool_engagement_stats["integrity_artifact_flags"]
    assert "execution_contract_asymmetry" in comparison.tool_engagement_stats["integrity_asymmetry_flags"]
    assert "read_only_loop_recovery_asymmetry" in comparison.tool_engagement_stats["integrity_asymmetry_flags"]
    assert comparison.tool_engagement_stats["baseline_loop_recovery_triggers"] == 1
    assert comparison.tool_engagement_stats["tok_loop_recovery_triggers"] == 0


def test_compare_pair_marks_loop_recovery_asymmetry_below_materiality_as_advisory(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )

    comparison = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(
            condition="baseline",
            family="execution_patch",
            success=True,
            grounding_success=True,
            tool_calls=3,
            eval_details={"command_invoked": True, "execution_contract_met": True},
            run_notes=("read_only_loop_recovery_step_3",),
        ),
        candidate=_task_run_result(
            condition="tok-universal",
            family="execution_patch",
            success=True,
            grounding_success=True,
            tool_calls=3,
            eval_details={"command_invoked": True, "execution_contract_met": True},
            run_notes=tuple(),
        ),
    )

    assert comparison.quality_gate_passed is True
    assert comparison.tool_engagement_stats["decision_grade"] is True
    assert comparison.tool_engagement_stats["loop_recovery_asymmetry_material"] is False
    assert comparison.tool_engagement_stats["loop_recovery_asymmetry_delta"] == 1
    assert comparison.tool_engagement_stats["decision_grade_advisories"] == [
        "read_only_loop_recovery_asymmetry_below_materiality"
    ]
    assert comparison.tool_engagement_stats["decision_grade_blockers"] == []


def test_compare_pair_blocks_material_loop_recovery_asymmetry_for_matched_pair(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path)
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": False},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"allowed_paths": ["src/app.py"], "visible_tests": []},
        }
    )

    comparison = evaluator.compare_pair(
        task=task,
        lane_id="production_claude_lane",
        repeat_index=1,
        baseline=_task_run_result(
            condition="baseline",
            family="execution_patch",
            success=True,
            grounding_success=True,
            tool_calls=4,
            eval_details={"command_invoked": True, "execution_contract_met": True},
            run_notes=(
                "read_only_loop_recovery_step_3",
                "read_only_loop_recovery_step_6",
                "read_only_loop_recovery_step_9",
            ),
        ),
        candidate=_task_run_result(
            condition="tok-universal",
            family="execution_patch",
            success=True,
            grounding_success=True,
            tool_calls=3,
            eval_details={"command_invoked": True, "execution_contract_met": True},
            run_notes=tuple(),
        ),
    )

    assert comparison.quality_gate_passed is True
    assert comparison.tool_engagement_stats["decision_grade"] is True
    assert comparison.tool_engagement_stats["loop_recovery_asymmetry_material"] is False
    assert comparison.tool_engagement_stats["loop_recovery_asymmetry_delta"] == 3
    assert (
        "read_only_loop_recovery_asymmetry_matched_pair"
        not in comparison.tool_engagement_stats["decision_grade_blockers"]
    )


def test_execution_failure_code_classifies_adapter_payload_contract_error() -> None:
    adapter_error = RuntimeError("'list' object has no attribute 'strip'")
    generic_error = RuntimeError("something else failed")

    assert _execution_failure_code(adapter_error) == "adapter_payload_contract_error"
    assert _execution_failure_code(generic_error) == "task_execution_failed"


def test_task_materializer_refuses_dirty_reportable_local_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pkg.py").write_text("VALUE = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "bench@example.test")
    _git(repo, "config", "user.name", "Bench Test")
    _git(repo, "add", "pkg.py")
    _git(repo, "commit", "-qm", "initial")
    (repo / "pkg.py").write_text("VALUE = 2\n")

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.dirty-checkout",
            "family": "execution_patch",
            "title": "Dirty checkout task",
            "summary": "Ensure dirty local checkouts are rejected for reportable runs.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the local bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": True},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["pkg.py"],
            "hidden_tests": ["tests/test_pkg.py::test_value"],
            "asset_dir": "assets/exec.local.dirty-checkout",
            "workspace_source": {"kind": "local_checkout", "path": str(repo)},
            "family_payload": {
                "allowed_paths": ["pkg.py"],
                "visible_tests": [],
                "hidden_tests": ["tests/test_pkg.py::test_value"],
            },
        }
    )

    materializer = TaskMaterializer(catalog_root=tmp_path, repo_root=repo)
    with pytest.raises(RuntimeError, match="clean checkout"):
        materializer.materialize(
            task,
            _production_lane(),
            repeat_index=1,
            condition="baseline",
            output_root=tmp_path / "reportable",
            reportable=True,
            local_debug=False,
        )

    materialized = materializer.materialize(
        task,
        _production_lane(),
        repeat_index=1,
        condition="baseline",
        output_root=tmp_path / "local_debug",
        reportable=False,
        local_debug=True,
    )
    assert Path(materialized.workspace_root).exists()


def test_task_materializer_applies_seed_patch(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    asset_root = catalog_root / "assets" / "seed-task"
    workspace = asset_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "module.py").write_text("VALUE = 1\n")
    _git(workspace, "init", "-q")
    _git(workspace, "add", "module.py")
    (workspace / "module.py").write_text("VALUE = 2\n")
    patch = subprocess.run(
        ["git", "diff"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (asset_root / "seed.patch").write_text(patch)
    (workspace / "module.py").write_text("VALUE = 1\n")
    (asset_root / "asset.lock.json").write_text(
        json.dumps(
            {
                "task_id": "exec.seed.patch",
                "workspace_sha256": _directory_sha256(workspace),
            }
        )
    )

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.seed.patch",
            "family": "execution_patch",
            "title": "Seed patch task",
            "summary": "Apply a failing-state patch before running.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Fix the seeded bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": True},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["module.py"],
            "hidden_tests": ["tests/test_module.py::test_value"],
            "asset_dir": str(asset_root.relative_to(catalog_root)),
            "workspace_source": {"kind": "asset_snapshot", "path": str(workspace.relative_to(catalog_root))},
            "seed_patch": "seed.patch",
            "family_payload": {
                "allowed_paths": ["module.py"],
                "visible_tests": [],
                "seed_patch_path": "seed.patch",
            },
        }
    )

    materialized = TaskMaterializer(catalog_root=catalog_root, repo_root=tmp_path).materialize(
        task,
        _production_lane(),
        repeat_index=1,
        condition="baseline",
        output_root=tmp_path / "out",
        reportable=True,
        local_debug=False,
    )
    assert (Path(materialized.workspace_root) / "module.py").read_text() == "VALUE = 2\n"


def test_task_materializer_bootstraps_pip_for_python_module_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_root = tmp_path / "catalog"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[build-system]\nrequires = []\nbuild-backend = "setuptools.build_meta"\n')
    (repo / "module.py").write_text("VALUE = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "bench@example.test")
    _git(repo, "config", "user.name", "Bench Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "baseline")

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.setup.pip-bootstrap",
            "family": "execution_patch",
            "title": "Bootstrap pip for setup",
            "summary": "Ensure python -m pip setup steps can run when pip is initially missing.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "python -m pip install -e . --no-deps",
            "prompt": "Fix the setup issue.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": True},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["module.py"],
            "hidden_tests": ["tests/test_module.py::test_value"],
            "asset_dir": "assets/exec.setup.pip-bootstrap",
            "workspace_source": {"kind": "local_checkout", "path": str(repo)},
            "family_payload": {
                "allowed_paths": ["module.py"],
                "visible_tests": [],
                "hidden_tests": ["tests/test_module.py::test_value"],
            },
        }
    )

    calls: list[str] = []

    def _fake_run_shell(
        command: str, *, cwd: Path, timeout_seconds: int, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout_seconds, extra_env
        calls.append(command)
        if command == "python -m pip --version":
            return subprocess.CompletedProcess(command, 1, "", "No module named pip")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    materializer = TaskMaterializer(catalog_root=catalog_root, repo_root=tmp_path)
    monkeypatch.setattr(materializer, "_run_shell", _fake_run_shell)

    materialized = materializer.materialize(
        task,
        _production_lane(),
        repeat_index=1,
        condition="baseline",
        output_root=tmp_path / "out",
        reportable=True,
        local_debug=False,
    )

    assert Path(materialized.workspace_root).exists()
    assert calls == [
        "python -m pip --version",
        "python -m ensurepip --upgrade",
        "python -m pip install --upgrade pip --quiet",
        "python -m pip install -e . --no-deps",
    ]


def test_family_evaluators_are_deterministic(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator()
    lane = _production_lane()

    grounding_task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.answer",
            "family": "repo_grounding",
            "title": "Grounded answer",
            "summary": "Answer with evidence.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Where is the answer defined?",
            "allowed_tools": ["view_file", "grep_search"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "repo_grounding",
                "min_grounded_retrieval_steps": 2,
                "requires_evidence_block": True,
            },
            "artifact_policy": {"publish_answer": True},
            "public_release": False,
            "asset_dir": "assets/qa.local.answer",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["src/app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [
                {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                {"file": "tests/test_app.py", "anchor": "test_answer_symbol", "why": "test"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    grounding_materialized = MaterializedBenchmarkTask(
        task=grounding_task,
        lane=lane,
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(tmp_path),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    grounding_answer = (
        "The answer lives in src/app.py and is exposed by answer_symbol.\n"
        "Evidence:\n"
        "- src/app.py :: def answer_symbol\n"
        "- tests/test_app.py :: test_answer_symbol"
    )
    first_grounding = evaluator.evaluate(
        grounding_materialized,
        answer_text=grounding_answer,
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[],
        workspace_root=tmp_path,
    )
    second_grounding = evaluator.evaluate(
        grounding_materialized,
        answer_text=grounding_answer,
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[],
        workspace_root=tmp_path,
    )
    assert first_grounding.to_dict() == second_grounding.to_dict()

    exec_workspace = tmp_path / "exec_ws"
    (exec_workspace / "src").mkdir(parents=True)
    (exec_workspace / "tests").mkdir(parents=True)
    (exec_workspace / "src" / "app.py").write_text("def answer():\n    return 1\n")
    (exec_workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(exec_workspace, "init", "-q")
    _git(exec_workspace, "config", "user.email", "bench@example.test")
    _git(exec_workspace, "config", "user.name", "Bench Test")
    _git(exec_workspace, "add", ".")
    _git(exec_workspace, "commit", "-qm", "baseline")
    (exec_workspace / "src" / "app.py").write_text("def answer():\n    return 2\n")
    exec_task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.patch",
            "family": "execution_patch",
            "title": "Patch task",
            "summary": "Fix the value bug.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "execution_patch",
                "clean_exit_required": True,
                "hidden_tests_timeout_seconds": 30,
            },
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": ["tests/test_app.py::test_answer"],
            "asset_dir": "assets/exec.local.patch",
            "workspace_source": {"kind": "local_checkout", "path": str(exec_workspace)},
            "family_payload": {
                "allowed_paths": ["src/app.py"],
                "visible_tests": [],
                "hidden_tests": ["tests/test_app.py::test_answer"],
            },
        }
    )
    exec_materialized = MaterializedBenchmarkTask(
        task=exec_task,
        lane=lane,
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(exec_workspace),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    first_exec = evaluator.evaluate(
        exec_materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[],
        workspace_root=exec_workspace,
    )
    second_exec = evaluator.evaluate(
        exec_materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[],
        workspace_root=exec_workspace,
    )
    assert first_exec.to_dict() == second_exec.to_dict()

    session_task = BenchmarkTaskManifest.from_dict(
        {
            "id": "session.local.answer",
            "family": "real_session",
            "title": "Session answer",
            "summary": "Reach the grounded milestone.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Continue the milestone.",
            "allowed_tools": ["view_file", "grep_search"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "real_session",
                "milestone_type": "grounded_answer",
                "invalid_tool_calls_limit": 0,
                "min_grounded_retrieval_steps": 2,
            },
            "artifact_policy": {"publish_summary": True},
            "public_release": False,
            "asset_dir": "assets/session.local.answer",
            "workspace_source": {"kind": "local_checkout", "path": "."},
            "family_payload": {
                "episode_bundle": {
                    "capture_source": "capture-review:candidate=context",
                    "redaction_version": "pilot-v1",
                    "episode_type": "grounded_answer",
                    "next_milestone": "Answer the question with evidence.",
                },
                "milestone_evaluator": {
                    "kind": "real_session",
                    "milestone_type": "grounded_answer",
                    "invalid_tool_calls_limit": 0,
                    "min_grounded_retrieval_steps": 2,
                },
            },
            "capture_source": "capture-review:candidate=context",
            "redaction_version": "pilot-v1",
            "episode_type": "grounded_answer",
            "next_milestone": "Answer the question with evidence.",
            "required_files": ["src/app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [
                {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                {"file": "tests/test_app.py", "anchor": "test_answer_symbol", "why": "test"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    session_materialized = MaterializedBenchmarkTask(
        task=session_task,
        lane=lane,
        repeat_index=1,
        condition="baseline",
        asset_root=str(tmp_path),
        workspace_root=str(tmp_path),
        resolved_ref="HEAD",
        reportable=False,
        setup_ran=False,
    )
    first_session = evaluator.evaluate(
        session_materialized,
        answer_text=grounding_answer,
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[],
        workspace_root=tmp_path,
    )
    second_session = evaluator.evaluate(
        session_materialized,
        answer_text=grounding_answer,
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=2,
        tool_records=[],
        workspace_root=tmp_path,
    )
    assert first_session.to_dict() == second_session.to_dict()


def test_run_catalog_benchmark_suite_pairs_baseline_and_tok(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    workspace = catalog_root / "assets" / "qa.local.answer" / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer_symbol():\n    return 'ok'\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer_symbol\n\n\ndef test_answer_symbol():\n    assert answer_symbol() == 'ok'\n"
    )
    asset_root = catalog_root / "assets" / "qa.local.answer"
    (asset_root / "gold_answer.json").write_text(
        json.dumps(
            {
                "required_files": ["src/app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [
                    {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                    {"file": "tests/test_app.py", "anchor": "test_answer_symbol", "why": "test"},
                ],
            }
        )
    )
    (asset_root / "asset.lock.json").write_text(
        json.dumps(
            {
                "task_id": "qa.local.answer",
                "workspace_sha256": _directory_sha256(workspace),
            }
        )
    )

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.answer",
            "family": "repo_grounding",
            "title": "Grounded local answer",
            "summary": "Answer with evidence from a tiny local repo.",
            "repo": "example/repo",
            "ref": "1234abcd",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined and tested?",
            "allowed_tools": ["grep_search", "view_file"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "repo_grounding",
                "min_grounded_retrieval_steps": 2,
                "requires_evidence_block": True,
            },
            "artifact_policy": {"publish_answer": True},
            "public_release": True,
            "asset_dir": "assets/qa.local.answer",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/qa.local.answer/workspace"},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["src/app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [
                {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                {"file": "tests/test_app.py", "anchor": "test_answer_symbol", "why": "test"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    catalog = BenchmarkCatalog(root=str(catalog_root), lanes=(_production_lane(),), tasks=(task,))

    responses = [
        "@Tool grep_search\n  search_path: src\n  query: answer_symbol\n",
        "@Tool view_file\n  path: src/app.py\n",
        "The implementation is in src/app.py and the symbol is answer_symbol.\nEvidence:\n- src/app.py :: def answer_symbol\n- tests/test_app.py :: test_answer_symbol",
        "@Tool grep_search\n  search_path: src\n  query: answer_symbol\n",
        "@Tool view_file\n  path: src/app.py\n",
        "The implementation is in src/app.py and the symbol is answer_symbol.\nEvidence:\n- src/app.py :: def answer_symbol\n- tests/test_app.py :: test_answer_symbol",
    ]
    live_runner = LiveBenchmarkRunner(
        model="anthropic/claude-test",
        provider="anthropic",
        client=_FakeClient(responses),
        timeout=10.0,
        max_tokens=120,
    )

    result = run_catalog_benchmark_suite(
        catalog=catalog,
        lane_id="production_claude_lane",
        output_root=tmp_path / "out",
        repeats=1,
        families=("repo_grounding",),
        include_advisory=False,
        local_debug=False,
        runner=live_runner,
        repo_root=tmp_path,
    )

    assert result.report.headline_summary().sample_size == 1
    assert result.report.headline_summary().tok_success_rate == 1.0
    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "raw_runs.json").exists()
    assert (tmp_path / "out" / "tasks" / "qa.local.answer" / "repeat_1" / "compare.json").exists()
    baseline_run = json.loads(
        (tmp_path / "out" / "tasks" / "qa.local.answer" / "repeat_1" / "baseline" / "run.json").read_text()
    )
    tok_run = json.loads(
        (tmp_path / "out" / "tasks" / "qa.local.answer" / "repeat_1" / "tok-universal" / "run.json").read_text()
    )
    assert int(baseline_run["tool_calls"]) > 0
    assert int(tok_run["tool_calls"]) > 0
    assert baseline_run["evaluation"]["success"] is True
    assert tok_run["evaluation"]["success"] is True


def test_live_benchmark_cli_program_both_writes_separate_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    catalog = BenchmarkCatalog(root=str(tmp_path), lanes=(_production_lane(),), tasks=tuple())
    report = build_benchmark_report(
        catalog,
        [
            BenchmarkComparisonRun.from_dict(
                {
                    "lane_id": "production_claude_lane",
                    "task_id": "exec.click.option-precedence",
                    "family": "execution_patch",
                    "repeat_index": 1,
                    "public_release": True,
                    "baseline_success": True,
                    "tok_success": True,
                    "quality_gate_passed": True,
                    "total_token_delta": -100,
                    "latency_delta_ms": 5.0,
                    "reacquisition_events": 0,
                    "invalid_tool_calls": 0,
                    "paired_result_stable": True,
                }
            )
        ],
    )

    def _fake_catalog_run(**kwargs):
        output_root = kwargs["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
        return CatalogBenchmarkRun(
            lane_id="production_claude_lane",
            selected_task_ids=("exec.click.option-precedence",),
            runs=tuple(),
            report=report,
        )

    def _fake_replay_suite(**kwargs):
        replay_root = kwargs["output"]
        replay_root.mkdir(parents=True, exist_ok=True)
        (replay_root / "coding-loop-5_compare.md").write_text("# replay")

    monkeypatch.setattr("tok.testing.benchmark_suite.load_benchmark_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr("tok.testing.benchmark_executor.run_catalog_benchmark_suite", _fake_catalog_run)
    monkeypatch.setattr("tok.cli._dev._run_legacy_compare_suite", _fake_replay_suite)

    result = runner.invoke(
        app,
        [
            "dev",
            "live-benchmark",
            "--program",
            "both",
            "--catalog-root",
            str(BENCHMARK_ROOT),
            "--output",
            str(tmp_path / "combined"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "combined" / "catalog" / "report.md").exists()
    assert (tmp_path / "combined" / "replay" / "coding-loop-5_compare.md").exists()
    assert (tmp_path / "combined" / "summary.md").exists()


def test_live_benchmark_cli_program_catalog_writes_report(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    catalog = BenchmarkCatalog(root=str(tmp_path), lanes=(_production_lane(),), tasks=tuple())
    report = build_benchmark_report(
        catalog,
        [
            BenchmarkComparisonRun.from_dict(
                {
                    "lane_id": "production_claude_lane",
                    "task_id": "qa.click.option-precedence",
                    "family": "repo_grounding",
                    "repeat_index": 1,
                    "public_release": True,
                    "baseline_success": True,
                    "tok_success": True,
                    "quality_gate_passed": True,
                    "total_token_delta": -25,
                    "latency_delta_ms": 2.0,
                    "reacquisition_events": 0,
                    "invalid_tool_calls": 0,
                    "paired_result_stable": True,
                }
            )
        ],
    )

    def _fake_catalog_run(**kwargs):
        output_root = kwargs["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
        return CatalogBenchmarkRun(
            lane_id="production_claude_lane",
            selected_task_ids=("qa.click.option-precedence",),
            runs=tuple(),
            report=report,
        )

    monkeypatch.setattr("tok.testing.benchmark_suite.load_benchmark_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr("tok.testing.benchmark_executor.run_catalog_benchmark_suite", _fake_catalog_run)

    result = runner.invoke(
        app,
        [
            "dev",
            "live-benchmark",
            "--program",
            "catalog",
            "--catalog-root",
            str(BENCHMARK_ROOT),
            "--output",
            str(tmp_path / "catalog_only"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "catalog_only" / "report.json").exists()
    assert (tmp_path / "catalog_only" / "report.md").exists()


def test_catalog_suite_fails_fast_when_evaluator_spec_is_missing(tmp_path: Path) -> None:
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.public.missing-evaluator-spec",
            "family": "execution_patch",
            "title": "Evaluator required",
            "summary": "Public execution tasks require a checked-in evaluator spec.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "execution_patch",
                "clean_exit_required": True,
                "evaluator_spec": "evaluators/missing.json",
            },
            "artifact_policy": {"publish_diff": True},
            "public_release": True,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": [],
            "asset_dir": "assets/exec.public.missing-evaluator-spec",
            "workspace_source": {
                "kind": "asset_snapshot",
                "path": "assets/exec.public.missing-evaluator-spec/workspace",
            },
            "seed_patch": "seed.patch",
            "family_payload": {
                "allowed_paths": ["src/app.py"],
                "visible_tests": [],
                "seed_patch_path": "seed.patch",
            },
        }
    )
    catalog = BenchmarkCatalog(root=str(tmp_path / "catalog"), lanes=(_production_lane(),), tasks=(task,))
    live_runner = LiveBenchmarkRunner(
        model="anthropic/claude-test",
        provider="anthropic",
        client=_FakeClient(["done"]),
        timeout=10.0,
        max_tokens=120,
    )

    with pytest.raises(RuntimeError, match="references evaluator spec"):
        run_catalog_benchmark_suite(
            catalog=catalog,
            lane_id="production_claude_lane",
            output_root=tmp_path / "out",
            repeats=1,
            families=("execution_patch",),
            include_advisory=False,
            public_release_only=True,
            local_debug=False,
            runner=live_runner,
            repo_root=tmp_path,
        )


def test_catalog_suite_records_materialization_failures_and_continues(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    asset_root = catalog_root / "assets" / "qa.lock.mismatch"
    workspace = asset_root / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer_symbol():\n    return 'ok'\n")
    (workspace / "tests" / "test_app.py").write_text("def test_answer_symbol():\n    assert True\n")
    (asset_root / "gold_answer.json").write_text(
        json.dumps(
            {
                "required_files": ["src/app.py", "tests/test_app.py"],
                "required_symbols": ["answer_symbol"],
                "supporting_spans": [
                    {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                    {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
                ],
            }
        )
    )
    (asset_root / "asset.lock.json").write_text(
        json.dumps(
            {
                "task_id": "qa.lock.mismatch",
                "workspace_sha256": "deadbeef",
            }
        )
    )
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.lock.mismatch",
            "family": "repo_grounding",
            "title": "Asset lock mismatch",
            "summary": "Materialization should fail per condition but not abort suite.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Where is answer_symbol defined?",
            "allowed_tools": ["list_dir", "view_file", "grep_search"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "repo_grounding",
                "min_grounded_retrieval_steps": 2,
                "requires_evidence_block": True,
            },
            "artifact_policy": {"publish_answer": True},
            "public_release": True,
            "asset_dir": "assets/qa.lock.mismatch",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/qa.lock.mismatch/workspace"},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["src/app.py", "tests/test_app.py"],
            "required_symbols": ["answer_symbol"],
            "supporting_spans": [
                {"file": "src/app.py", "anchor": "def answer_symbol", "why": "implementation"},
                {"file": "tests/test_app.py", "anchor": "def test_answer_symbol", "why": "test"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )
    catalog = BenchmarkCatalog(root=str(catalog_root), lanes=(_production_lane(),), tasks=(task,))
    live_runner = LiveBenchmarkRunner(
        model="anthropic/claude-test",
        provider="anthropic",
        client=_FakeClient(["unused"]),
        timeout=10.0,
        max_tokens=120,
    )

    result = run_catalog_benchmark_suite(
        catalog=catalog,
        lane_id="production_claude_lane",
        output_root=tmp_path / "out",
        repeats=1,
        families=("repo_grounding",),
        include_advisory=False,
        public_release_only=True,
        local_debug=False,
        runner=live_runner,
        repo_root=tmp_path,
    )

    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.baseline_success is False
    assert run.tok_success is False
    assert run.quality_gate_passed is False
    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "raw_runs.json").exists()
    baseline_run = json.loads(
        (tmp_path / "out" / "tasks" / "qa.lock.mismatch" / "repeat_1" / "baseline" / "run.json").read_text()
    )
    tok_run = json.loads(
        (tmp_path / "out" / "tasks" / "qa.lock.mismatch" / "repeat_1" / "tok-universal" / "run.json").read_text()
    )
    assert baseline_run["local_failure"] == "asset_lock_hash_mismatch"
    assert tok_run["local_failure"] == "asset_lock_hash_mismatch"


def test_family_evaluator_uses_checked_in_evaluator_spec(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(catalog_root=tmp_path / "benchmarks")
    lane = _production_lane()
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("def answer():\n    return 2\n")
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "bench@example.test")
    _git(workspace, "config", "user.name", "Bench Test")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "baseline")

    evaluator_root = tmp_path / "benchmarks" / "evaluators"
    evaluator_root.mkdir(parents=True)
    (evaluator_root / "exec.public.overlay.json").write_text(
        json.dumps(
            {
                "selectors": ["tests/test_app.py::test_answer"],
                "timeout_seconds": 30,
            }
        )
    )

    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.public.overlay",
            "family": "execution_patch",
            "title": "Overlay evaluator task",
            "summary": "Use selectors from a checked-in evaluator spec.",
            "repo": "example/repo",
            "ref": "deadbeef",
            "setup_script": "no_setup_required",
            "prompt": "Fix the bug.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "execution_patch",
                "clean_exit_required": True,
                "evaluator_spec": "evaluators/exec.public.overlay.json",
            },
            "artifact_policy": {"publish_diff": True},
            "public_release": True,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": [],
            "asset_dir": "assets/exec.public.overlay",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/exec.public.overlay/workspace"},
            "seed_patch": "seed.patch",
            "family_payload": {
                "allowed_paths": ["src/app.py"],
                "visible_tests": [],
                "seed_patch_path": "seed.patch",
            },
        }
    )
    materialized = MaterializedBenchmarkTask(
        task=task,
        lane=lane,
        repeat_index=1,
        condition="tok-universal",
        asset_root=str(tmp_path / "assets" / "exec.public.overlay"),
        workspace_root=str(workspace),
        resolved_ref="deadbeef",
        reportable=True,
        setup_ran=False,
    )

    result = evaluator.evaluate(
        materialized,
        answer_text="done",
        clean_exit=True,
        invalid_tool_calls=0,
        tool_calls=1,
        tool_records=[
            ToolExecutionRecord(
                step_index=1,
                tool_name="edit_file",
                canonical_tool_name="edit_file",
                tool_input={"path": "src/app.py"},
                invalid=False,
                is_error=False,
                content_preview="Edited src/app.py",
            ),
            ToolExecutionRecord(
                step_index=2,
                tool_name="run_tests",
                canonical_tool_name="run_tests",
                tool_input={"command": "python -m pytest -q tests/test_app.py::test_answer"},
                invalid=False,
                is_error=False,
                content_preview="$ python -m pytest -q tests/test_app.py::test_answer",
            ),
        ],
        workspace_root=workspace,
    )

    assert result.success is True
    assert result.details["evaluator_spec"] == "evaluators/exec.public.overlay.json"


def test_live_benchmark_cli_program_both_defaults_catalog_repeats_to_five(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    catalog = BenchmarkCatalog(root=str(tmp_path), lanes=(_production_lane(),), tasks=tuple())
    report = build_benchmark_report(
        catalog,
        [
            BenchmarkComparisonRun.from_dict(
                {
                    "lane_id": "production_claude_lane",
                    "task_id": "qa.click.option-precedence",
                    "family": "repo_grounding",
                    "repeat_index": 1,
                    "public_release": True,
                    "baseline_success": True,
                    "tok_success": True,
                    "quality_gate_passed": True,
                    "total_token_delta": -25,
                    "latency_delta_ms": 2.0,
                    "reacquisition_events": 0,
                    "invalid_tool_calls": 0,
                    "paired_result_stable": True,
                }
            )
        ],
    )
    captured: dict[str, object] = {}

    def _fake_catalog_run(**kwargs):
        captured["catalog"] = kwargs
        output_root = kwargs["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
        return CatalogBenchmarkRun(
            lane_id="production_claude_lane",
            selected_task_ids=("qa.click.option-precedence",),
            runs=tuple(),
            report=report,
        )

    def _fake_replay_suite(**kwargs):
        captured["replay"] = kwargs
        replay_root = kwargs["output"]
        replay_root.mkdir(parents=True, exist_ok=True)
        (replay_root / "coding-loop-5_compare.md").write_text("# replay")

    monkeypatch.setattr("tok.testing.benchmark_suite.load_benchmark_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr("tok.testing.benchmark_executor.run_catalog_benchmark_suite", _fake_catalog_run)
    monkeypatch.setattr("tok.cli._dev._run_legacy_compare_suite", _fake_replay_suite)

    result = runner.invoke(
        app,
        [
            "dev",
            "live-benchmark",
            "--program",
            "both",
            "--catalog-root",
            str(BENCHMARK_ROOT),
            "--output",
            str(tmp_path / "combined"),
        ],
    )

    assert result.exit_code == 0
    assert captured["catalog"]["repeats"] == 5
    assert captured["replay"]["repeats"] == 5
