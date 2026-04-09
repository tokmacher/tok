from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tok.cli import app
from tok.testing.benchmark_executor import (
    CatalogBenchmarkRun,
    FamilyEvaluator,
    MaterializedBenchmarkTask,
    TaskMaterializer,
    _directory_sha256,
    run_catalog_benchmark_suite,
)
from tok.testing.benchmark_suite import (
    BenchmarkCatalog,
    BenchmarkComparisonRun,
    BenchmarkLane,
    BenchmarkTaskManifest,
    build_benchmark_report,
    load_benchmark_catalog,
)
from tok.testing.live_benchmark import LiveBenchmarkRunner

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


def test_live_benchmark_cli_program_both_writes_separate_artifacts(monkeypatch, tmp_path: Path) -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
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

    def _fake_legacy_suite(**kwargs):
        legacy_root = kwargs["output"]
        legacy_root.mkdir(parents=True, exist_ok=True)
        (legacy_root / "coding-loop-5_compare.md").write_text("# legacy")

    monkeypatch.setattr("tok.testing.benchmark_executor.run_catalog_benchmark_suite", _fake_catalog_run)
    monkeypatch.setattr("tok.cli._dev._run_legacy_compare_suite", _fake_legacy_suite)

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
    assert (tmp_path / "combined" / "legacy" / "coding-loop-5_compare.md").exists()
    assert (tmp_path / "combined" / "summary.md").exists()


def test_live_benchmark_cli_program_catalog_writes_report(monkeypatch, tmp_path: Path) -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
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


def test_catalog_suite_fails_fast_when_private_evaluator_overlay_is_missing(tmp_path: Path) -> None:
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.public.private-overlay",
            "family": "execution_patch",
            "title": "Overlay required",
            "summary": "Public execution tasks require a private evaluator overlay.",
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
                "hidden_evaluator_ref": "exec.public.private-overlay",
            },
            "artifact_policy": {"publish_diff": True},
            "public_release": True,
            "allowed_paths": ["src/app.py"],
            "hidden_tests": [],
            "asset_dir": "assets/exec.public.private-overlay",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/exec.public.private-overlay/workspace"},
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

    with pytest.raises(RuntimeError, match="requires private evaluator"):
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
            private_evaluator_root=None,
        )


def test_family_evaluator_uses_private_hidden_evaluator_overlay(tmp_path: Path) -> None:
    evaluator = FamilyEvaluator(private_evaluator_root=tmp_path / "private")
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

    private_root = tmp_path / "private"
    private_root.mkdir()
    (private_root / "exec.public.overlay.json").write_text(
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
            "summary": "Use selectors from the private evaluator overlay.",
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
                "hidden_evaluator_ref": "exec.public.overlay",
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
        tool_records=[],
        workspace_root=workspace,
    )

    assert result.success is True
    assert result.details["hidden_evaluator_ref"] == "exec.public.overlay"


def test_live_benchmark_cli_program_both_defaults_catalog_repeats_to_three(monkeypatch, tmp_path: Path) -> None:
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
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

    def _fake_legacy_suite(**kwargs):
        captured["legacy"] = kwargs
        legacy_root = kwargs["output"]
        legacy_root.mkdir(parents=True, exist_ok=True)
        (legacy_root / "coding-loop-5_compare.md").write_text("# legacy")

    monkeypatch.setattr("tok.testing.benchmark_executor.run_catalog_benchmark_suite", _fake_catalog_run)
    monkeypatch.setattr("tok.cli._dev._run_legacy_compare_suite", _fake_legacy_suite)

    private_root = tmp_path / "private"
    private_root.mkdir()
    result = runner.invoke(
        app,
        [
            "dev",
            "live-benchmark",
            "--program",
            "both",
            "--catalog-root",
            str(BENCHMARK_ROOT),
            "--private-evaluator-root",
            str(private_root),
            "--output",
            str(tmp_path / "combined"),
        ],
    )

    assert result.exit_code == 0
    assert captured["catalog"]["repeats"] == 3
    assert captured["legacy"]["repeats"] == 3
    assert captured["catalog"]["private_evaluator_root"] == private_root
