from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tok.testing.benchmark_suite import BenchmarkTaskManifest


def _load_module() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_benchmark_assets.py"
    spec = importlib.util.spec_from_file_location("prepare_benchmark_assets", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _grounding_task() -> BenchmarkTaskManifest:
    return BenchmarkTaskManifest.from_dict(
        {
            "id": "qa.local.asset-build",
            "family": "repo_grounding",
            "title": "Local asset build",
            "summary": "Build a tiny repo-grounding asset from the local repository.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Where is the release surface defined?",
            "allowed_tools": ["list_dir", "grep_search", "view_file"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {
                "kind": "repo_grounding",
                "min_grounded_retrieval_steps": 2,
                "requires_evidence_block": True,
            },
            "artifact_policy": {"publish_answer": True},
            "public_release": True,
            "asset_dir": "assets/qa.local.asset-build",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/qa.local.asset-build/workspace"},
            "family_payload": {"gold_answer_path": "gold_answer.json"},
            "required_files": ["README.md", ".gitignore"],
            "required_symbols": ["tok", ".venv/"],
            "supporting_spans": [
                {"file": "README.md", "anchor": "# Tok", "why": "product README"},
                {"file": ".gitignore", "anchor": ".venv/", "why": "gitignore rule"},
            ],
            "answer_contract": "Answer in at most 6 sentences followed by an Evidence block with 2 to 4 citations.",
        }
    )


def _execution_task() -> BenchmarkTaskManifest:
    return BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.asset-build",
            "family": "execution_patch",
            "title": "Local execution asset build",
            "summary": "Generate a seed patch from a deterministic local edit.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the seeded benchmark README change.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": True},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "asset_dir": "assets/exec.local.asset-build",
            "workspace_source": {"kind": "asset_snapshot", "path": "assets/exec.local.asset-build/workspace"},
            "family_payload": {
                "allowed_paths": ["README.md"],
                "visible_tests": [],
                "hidden_tests": ["README.md::noop"],
                "seed_patch_path": "seed.patch",
                "seed_patch_edits": [
                    {
                        "path": "README.md",
                        "old_string": "Tok is a local Claude Code bridge",
                        "new_string": "Tok is a local Claude Code benchmark bridge",
                    }
                ],
            },
            "seed_patch": "seed.patch",
            "allowed_paths": ["README.md"],
            "hidden_tests": ["README.md::noop"],
        }
    )


def test_build_task_asset_writes_gold_answer_and_lock(tmp_path: Path) -> None:
    module = _load_module()
    task = _grounding_task()

    module.build_task_asset(task, root=tmp_path, force=False)

    asset_root = tmp_path / task.asset_dir
    workspace_root = asset_root / "workspace"
    gold_answer = json.loads((asset_root / "gold_answer.json").read_text())
    lock_payload = json.loads((asset_root / "asset.lock.json").read_text())

    assert (workspace_root / "README.md").exists()
    assert gold_answer["required_files"] == ["README.md", ".gitignore"]
    assert lock_payload["task_id"] == task.id
    assert lock_payload["workspace_sha256"] == module._workspace_sha256(workspace_root)


def test_build_task_asset_generates_seed_patch_from_declared_edits(tmp_path: Path) -> None:
    module = _load_module()
    task = _execution_task()

    module.build_task_asset(task, root=tmp_path, force=False)

    asset_root = tmp_path / task.asset_dir
    workspace_root = asset_root / "workspace"
    seed_patch = (asset_root / "seed.patch").read_text()
    workspace_text = (workspace_root / "README.md").read_text()

    assert "Tok is a local Claude Code benchmark bridge" in seed_patch
    assert "Tok is a local Claude Code bridge" in workspace_text


def test_build_seed_patch_preserves_trailing_blank_context_lines(tmp_path: Path) -> None:
    module = _load_module()
    task = BenchmarkTaskManifest.from_dict(
        {
            "id": "exec.local.trailing-blank-context",
            "family": "execution_patch",
            "title": "Trailing blank context",
            "summary": "Ensure seed patches remain applicable when a hunk ends on a blank context line.",
            "repo": "tok",
            "ref": "HEAD",
            "setup_script": "no_setup_required",
            "prompt": "Fix the synthetic trailing blank line edit.",
            "allowed_tools": ["view_file", "edit_file", "run_tests"],
            "time_budget_minutes": 1,
            "step_budget": 4,
            "success_evaluator": {"kind": "execution_patch", "clean_exit_required": True},
            "artifact_policy": {"publish_diff": True},
            "public_release": False,
            "asset_dir": "assets/exec.local.trailing-blank-context",
            "workspace_source": {
                "kind": "asset_snapshot",
                "path": "assets/exec.local.trailing-blank-context/workspace",
            },
            "family_payload": {
                "allowed_paths": ["sample.txt"],
                "visible_tests": [],
                "hidden_tests": ["sample.txt::noop"],
                "seed_patch_path": "seed.patch",
                "seed_patch_edits": [
                    {
                        "path": "sample.txt",
                        "old_string": "gamma",
                        "new_string": "delta",
                    }
                ],
            },
            "seed_patch": "seed.patch",
            "allowed_paths": ["sample.txt"],
            "hidden_tests": ["sample.txt::noop"],
        }
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "sample.txt").write_text("alpha\nbeta\ngamma\n\n")

    seed_patch = module._build_seed_patch(workspace_root, task)
    patch_path = tmp_path / "seed.patch"
    patch_path.write_text(seed_patch)

    subprocess.run(["git", "init", "-q"], cwd=workspace_root, check=True)
    completed = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path.resolve())],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_verify_assets_reports_workspace_hash_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    task = _grounding_task()

    module.build_task_asset(task, root=tmp_path, force=False)
    workspace_file = tmp_path / task.asset_dir / "workspace" / "README.md"
    workspace_file.write_text(workspace_file.read_text() + "\nextra drift\n")

    errors = module.verify_assets(tmp_path, task_ids=(), public_only=False, tasks=(task,))

    assert any("workspace hash mismatch" in error for error in errors)


def test_refresh_locks_recomputes_workspace_hash(tmp_path: Path) -> None:
    module = _load_module()
    task = _grounding_task()

    module.build_task_asset(task, root=tmp_path, force=False)
    workspace_file = tmp_path / task.asset_dir / "workspace" / "README.md"
    workspace_file.write_text(workspace_file.read_text() + "\nrefreshed\n")

    module.refresh_locks(tmp_path, task_ids=(), public_only=False, tasks=(task,))
    errors = module.verify_assets(tmp_path, task_ids=(), public_only=False, tasks=(task,))

    assert errors == []
