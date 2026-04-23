from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tok.testing.benchmark_suite import BenchmarkLane, BenchmarkTaskManifest

from ._models import ASSET_LOCK_FILENAME, MaterializedBenchmarkTask
from ._utils import (
    _copytree_ignore,
    _directory_sha256,
    _git,
    _resolved_path,
    _run_zsh_command,
)


class TaskMaterializer:
    """Create isolated workspaces for benchmark tasks."""

    def __init__(self, *, catalog_root: Path, repo_root: Path | None = None) -> None:
        self.catalog_root = catalog_root.resolve()
        self.repo_root = (repo_root or Path.cwd()).resolve()

    def materialize(
        self,
        task: BenchmarkTaskManifest,
        lane: BenchmarkLane,
        *,
        repeat_index: int,
        condition: str,
        output_root: Path,
        reportable: bool,
        local_debug: bool,
    ) -> MaterializedBenchmarkTask:
        asset_root = self._asset_root(task)
        source_spec = task.effective_workspace_source()
        source_path = self._workspace_source_path(source_spec)
        if not source_path.exists():
            msg = f"workspace source not found for {task.id}: {source_path}"
            raise FileNotFoundError(msg)

        if reportable and source_spec.get("kind") == "asset_snapshot":
            self._validate_reportable_asset(task, asset_root=asset_root, workspace_root=source_path)

        resolved_ref = self._resolved_ref(task, source_spec, source_path)
        if reportable and not local_debug and source_spec.get("kind") == "local_checkout":
            self._assert_clean_checkout(source_path)

        workspace_root = output_root / "workspace"
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
        shutil.copytree(source_path, workspace_root, ignore=_copytree_ignore)
        _git(["init", "-q"], cwd=workspace_root, check=True)
        _git(["config", "user.email", "benchmark@example.test"], cwd=workspace_root, check=True)
        _git(["config", "user.name", "Tok Benchmark"], cwd=workspace_root, check=True)

        if task.seed_patch:
            self._apply_seed_patch(task, workspace_root, asset_root)

        _git(["add", "-A"], cwd=workspace_root, check=True)
        _git(["commit", "-qm", "benchmark baseline", "--allow-empty"], cwd=workspace_root, check=False)

        self._ensure_build_stubs(task, workspace_root)

        setup_ran = False
        if task.setup_script.strip() and task.setup_script.strip() != "no_setup_required":
            self._prepare_setup_environment(
                task.setup_script,
                cwd=workspace_root,
                timeout_seconds=max(60, task.time_budget_minutes * 60),
            )
            completed = self._run_shell(
                task.setup_script,
                cwd=workspace_root,
                timeout_seconds=max(60, task.time_budget_minutes * 60),
                extra_env=self._setup_env_overrides(task),
            )
            if completed.returncode != 0:
                msg = f"setup_script failed for {task.id}: {completed.stderr.strip() or completed.stdout.strip()}"
                raise RuntimeError(msg)
            setup_ran = True

        return MaterializedBenchmarkTask(
            task=task,
            lane=lane,
            repeat_index=repeat_index,
            condition=condition,
            asset_root=str(asset_root),
            workspace_root=str(workspace_root),
            resolved_ref=resolved_ref,
            reportable=reportable,
            setup_ran=setup_ran,
        )

    def _prepare_setup_environment(self, setup_script: str, *, cwd: Path, timeout_seconds: int) -> None:
        normalized = setup_script.strip()
        if "python -m pip" not in normalized:
            return
        probe = self._run_shell(
            "python -m pip --version",
            cwd=cwd,
            timeout_seconds=min(timeout_seconds, 30),
        )
        if probe.returncode != 0:
            bootstrap = self._run_shell(
                "python -m ensurepip --upgrade",
                cwd=cwd,
                timeout_seconds=min(timeout_seconds, 60),
            )
            if bootstrap.returncode != 0:
                msg = bootstrap.stderr.strip() or bootstrap.stdout.strip() or "unknown ensurepip failure"
                raise RuntimeError(f"setup bootstrap failed: {msg}")
        self._run_shell(
            "python -m pip install --upgrade pip --quiet",
            cwd=cwd,
            timeout_seconds=min(timeout_seconds, 60),
        )

    _STUB_FILES: dict[str, str | None] = {
        "CHANGELOG.md": "# Changelog\n\n## 24.2.0\n\n### Bugfixes\n\n- Stub for benchmark.\n\n## older\n",
        "changelog.d": None,
        "README.md": "# attrs\n",
    }

    def _ensure_build_stubs(self, task: BenchmarkTaskManifest, workspace_root: Path) -> None:
        if "pip install" not in task.setup_script:
            return
        pyproject = workspace_root / "pyproject.toml"
        if not pyproject.exists():
            return
        try:
            text = pyproject.read_text()
        except OSError:
            return
        for name, content in self._STUB_FILES.items():
            if name not in text:
                continue
            target = workspace_root / name
            if target.exists():
                continue
            if content is not None:
                target.write_text(content)
            else:
                target.mkdir(parents=True, exist_ok=True)

    def _setup_env_overrides(self, task: BenchmarkTaskManifest) -> dict[str, str]:
        overrides: dict[str, str] = {}
        ref = task.ref.strip()
        if ref:
            overrides["SETUPTOOLS_SCM_PRETEND_VERSION"] = ref
            overrides["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ATTRS"] = ref
        return overrides

    def _asset_root(self, task: BenchmarkTaskManifest) -> Path:
        raw_asset_dir = task.effective_asset_dir()
        asset_dir = Path(raw_asset_dir)
        if asset_dir.is_absolute():
            return asset_dir
        return (self.catalog_root / raw_asset_dir).resolve()

    def _workspace_source_path(self, source_spec: dict[str, Any]) -> Path:
        kind = str(source_spec.get("kind") or "asset_snapshot").strip() or "asset_snapshot"
        raw_path = str(source_spec.get("path") or "").strip()
        if not raw_path:
            msg = "workspace_source requires a path"
            raise ValueError(msg)
        base = self.repo_root if kind == "local_checkout" else self.catalog_root
        return _resolved_path(base, raw_path)

    def _resolved_ref(self, task: BenchmarkTaskManifest, source_spec: dict[str, Any], source_path: Path) -> str:
        if str(source_spec.get("kind") or "") == "local_checkout":
            completed = _git(["rev-parse", task.ref], cwd=source_path)
            if completed.returncode == 0:
                return completed.stdout.strip()
        return task.ref

    def _assert_clean_checkout(self, source_path: Path) -> None:
        status = _git(["status", "--porcelain"], cwd=source_path)
        if status.returncode != 0:
            msg = f"unable to inspect checkout state: {status.stderr.strip()}"
            raise RuntimeError(msg)
        if status.stdout.strip():
            msg = "reportable benchmark runs require a clean checkout; rerun with local-debug enabled"
            raise RuntimeError(msg)

    def _validate_reportable_asset(
        self,
        task: BenchmarkTaskManifest,
        *,
        asset_root: Path,
        workspace_root: Path,
    ) -> None:
        lock_path = asset_root / ASSET_LOCK_FILENAME
        if not lock_path.exists():
            msg = f"reportable asset lock missing for {task.id}: {lock_path}"
            raise RuntimeError(msg)
        lock_payload = json.loads(lock_path.read_text())
        if str(lock_payload.get("task_id", "")).strip() not in {"", task.id}:
            msg = f"asset lock task_id mismatch for {task.id}: {lock_payload.get('task_id')}"
            raise RuntimeError(msg)
        recorded_hash = str(lock_payload.get("workspace_sha256", "")).strip()
        if not recorded_hash:
            msg = f"asset lock missing workspace_sha256 for {task.id}: {lock_path}"
            raise RuntimeError(msg)
        actual_hash = _directory_sha256(workspace_root)
        if actual_hash != recorded_hash:
            msg = f"asset lock hash mismatch for {task.id}: expected {recorded_hash} got {actual_hash}"
            raise RuntimeError(msg)
        if task.family == "execution_patch":
            seed_patch = asset_root / task.seed_patch
            if not seed_patch.exists():
                msg = f"seed patch missing for {task.id}: {seed_patch}"
                raise RuntimeError(msg)
        if task.family == "repo_grounding":
            gold_answer_path = str(task.family_payload.get("gold_answer_path") or "").strip()
            if not gold_answer_path:
                msg = f"gold_answer_path missing for {task.id}"
                raise RuntimeError(msg)
            gold_answer = asset_root / gold_answer_path
            if not gold_answer.exists():
                msg = f"gold answer missing for {task.id}: {gold_answer}"
                raise RuntimeError(msg)

    def _apply_seed_patch(self, task: BenchmarkTaskManifest, workspace_root: Path, asset_root: Path) -> None:
        patch_source = Path(task.seed_patch)
        if not patch_source.is_absolute():
            candidate = (asset_root / task.seed_patch).resolve()
            if candidate.exists():
                patch_source = candidate
        patch_path: Path
        cleanup_path = False
        if patch_source.exists():
            patch_path = patch_source
        else:
            patch_path = workspace_root / "__seed_patch__.diff"
            patch_path.write_text(task.seed_patch)
            cleanup_path = True
        try:
            completed = _git(["apply", "--whitespace=nowarn", str(patch_path)], cwd=workspace_root)
            if completed.returncode != 0:
                msg = completed.stderr.strip() or completed.stdout.strip() or "unknown git apply failure"
                raise RuntimeError(f"seed patch failed for {task.id}: {msg}")
        finally:
            if cleanup_path and patch_path.exists():
                patch_path.unlink()

    def _run_shell(
        self, command: str, *, cwd: Path, timeout_seconds: int, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if extra_env:
            env.update(extra_env)
        return _run_zsh_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            env=env,
        ).as_completed_process()
