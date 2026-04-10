"""Full executor for production benchmark catalog tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tok.gateway import BridgeSession
from tok.runtime.core import RuntimeSession
from tok.testing.benchmark_suite import (
    BenchmarkCatalog,
    BenchmarkComparisonRun,
    BenchmarkLane,
    BenchmarkReport,
    BenchmarkTaskManifest,
    build_benchmark_report,
)
from tok.testing.live_benchmark import LiveBenchmarkRunner
from tok.testing.stress.executor import ReadOnlyToolExecutor

_SKIP_COPY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".ruff_cache",
    ".venv",
    "tmp",
}
ASSET_LOCK_FILENAME = "asset.lock.json"
DEFAULT_EVALUATOR_BUNDLE_DIR = "evaluators"
_TEST_COMMAND_RE = re.compile(r"^(?:uv run )?(?:python -m )?pytest\b")


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _copytree_ignore(_src: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _SKIP_COPY_NAMES}


def _resolved_path(base: Path, raw_path: str | None) -> Path:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return base
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel_path = path.relative_to(root).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(args: list[str], *, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _safe_relative_path(workspace_root: Path, candidate: Path) -> str:
    try:
        return str(candidate.relative_to(workspace_root))
    except ValueError:
        return str(candidate)


def _assistant_message_content(blocks: list[dict[str, Any]], fallback_text: str) -> str | list[dict[str, Any]]:
    if not blocks:
        return fallback_text
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text", ""))
    return copy.deepcopy(blocks)


def _count_reacquisition_events(signals: dict[str, int]) -> int:
    event_keys = (
        "answer_anchor_reacquisition_attempt",
        "answer_ready_reacquisition_attempt",
        "repair_phase_reacquisition_attempt",
        "validated_target_reacquisition_attempt",
        "validated_target_exact_reacquisition_attempt",
        "validated_target_reconfirmation_attempt",
    )
    count = sum(int(signals.get(key, 0)) for key in event_keys)
    if count:
        return count
    return 1 if int(signals.get("reacquisition_cost_tokens", 0)) > 0 else 0


def _file_mentioned(path: str, text: str) -> bool:
    if path in text:
        return True
    basename = Path(path).name
    if basename in text and basename != path:
        return True
    return False


def _sentence_count(text: str) -> int:
    body = text.split("Evidence:", 1)[0].strip()
    if not body:
        return 0
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", body) if piece.strip()]
    return len(pieces)


_THINKING_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def _clean_answer_text(text: str) -> str:
    return _THINKING_TOKEN_RE.sub("", text).strip()


_EVIDENCE_NOISE_RE = re.compile(r"^\*+\s*$")


def _evidence_lines(text: str) -> list[str]:
    if "Evidence:" not in text:
        return []
    _, evidence = text.split("Evidence:", 1)
    lines: list[str] = []
    for raw_line in evidence.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _EVIDENCE_NOISE_RE.match(line):
            continue
        if line.startswith("-"):
            lines.append(line.lstrip("-").strip())
        else:
            lines.append(line)
    return lines


def _normalize_pytest_output(text: str) -> str:
    return re.sub(r"in \d+(?:\.\d+)?s", "in <time>s", text)


@dataclass(frozen=True)
class MaterializedBenchmarkTask:
    task: BenchmarkTaskManifest
    lane: BenchmarkLane
    repeat_index: int
    condition: str
    asset_root: str
    workspace_root: str
    resolved_ref: str
    reportable: bool
    setup_ran: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolExecutionRecord:
    step_index: int
    tool_name: str
    canonical_tool_name: str
    tool_input: dict[str, Any]
    invalid: bool
    is_error: bool
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskEvaluationResult:
    success: bool
    grounding_success: bool
    details: dict[str, Any]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "grounding_success": self.grounding_success,
            "details": dict(self.details),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkTaskRunResult:
    lane_id: str
    condition: str
    task_id: str
    family: str
    repeat_index: int
    workspace_root: str
    answer_text: str
    raw_response: str
    provider_usage: dict[str, Any]
    tool_calls: int
    invalid_tool_calls: int
    reacquisition_events: int
    clean_exit: bool
    modified_files: tuple[str, ...]
    tool_records: tuple[ToolExecutionRecord, ...]
    turns: tuple[dict[str, Any], ...]
    evaluation: TaskEvaluationResult
    local_failure: str = ""
    notes: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.evaluation.success

    @property
    def grounding_success(self) -> bool:
        return self.evaluation.grounding_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "family": self.family,
            "repeat_index": self.repeat_index,
            "workspace_root": self.workspace_root,
            "answer_text": self.answer_text,
            "raw_response": self.raw_response,
            "provider_usage": dict(self.provider_usage),
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "reacquisition_events": self.reacquisition_events,
            "clean_exit": self.clean_exit,
            "modified_files": list(self.modified_files),
            "tool_records": [record.to_dict() for record in self.tool_records],
            "turns": [dict(turn) for turn in self.turns],
            "evaluation": self.evaluation.to_dict(),
            "local_failure": self.local_failure,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CatalogBenchmarkRun:
    lane_id: str
    selected_task_ids: tuple[str, ...]
    runs: tuple[BenchmarkComparisonRun, ...]
    report: BenchmarkReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "selected_task_ids": list(self.selected_task_ids),
            "runs": [run.to_dict() for run in self.runs],
            "report": self.report.to_dict(),
        }


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
        return subprocess.run(
            command,
            cwd=cwd,
            shell=False,
            executable="/bin/zsh",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )


class BenchmarkToolExecutor:
    """Execute the benchmark tool allowlist inside a task workspace."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        allowed_tools: tuple[str, ...],
        timeout_seconds: int = 120,
        max_output_chars: int = 12000,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.allowed_tools = tuple(allowed_tools)
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.read_only = ReadOnlyToolExecutor(self.workspace_root, max_output_chars=max_output_chars)

    def execute_tool(
        self,
        block: dict[str, Any],
        *,
        step_index: int,
    ) -> tuple[dict[str, Any], bool, ToolExecutionRecord]:
        tool_input = block.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        canonical_name = self._canonical_tool_name(str(block.get("name", "")).strip())
        if not canonical_name or canonical_name not in self.allowed_tools:
            result = self._tool_result_message(
                block,
                f"ERROR: disallowed tool `{block.get('name', '')}` for this benchmark task",
                is_error=True,
                signal="invalid_tool",
            )
            record = ToolExecutionRecord(
                step_index=step_index,
                tool_name=str(block.get("name", "")).strip(),
                canonical_tool_name=canonical_name or "unsupported",
                tool_input=dict(tool_input),
                invalid=True,
                is_error=True,
                content_preview=_truncate(str(result["content"])),
            )
            return result, True, record

        if canonical_name in {"view_file", "grep_search", "list_dir"}:
            read_only_result, invalid = self.read_only.execute(dict(block))
            signal = str(read_only_result.get("contract_signal", ""))
            invalid = invalid or signal in {"invalid_tool", "bad_tool_args", "mutating_tool", "unsupported_tool"}
            content = str(read_only_result.get("content", ""))
            if (
                canonical_name == "grep_search"
                and content
                and content != "(no matches)"
                and not read_only_result.get("is_error")
                and re.search(r":\d+[:-]", content)
            ):
                content += "\n\nHint: use view_file with start and end parameters to read around these matches."
            result = self._tool_result_message(
                block,
                content,
                is_error=bool(read_only_result.get("is_error")),
                signal=signal,
            )
            record = ToolExecutionRecord(
                step_index=step_index,
                tool_name=str(block.get("name", "")).strip(),
                canonical_tool_name=canonical_name,
                tool_input=dict(tool_input),
                invalid=invalid,
                is_error=bool(read_only_result.get("is_error")),
                content_preview=_truncate(str(read_only_result.get("content", ""))),
            )
            return result, invalid, record

        if canonical_name == "edit_file":
            content, is_error, signal = self._edit_file(tool_input)
        elif canonical_name == "run_tests":
            content, is_error, signal = self._run_tests(tool_input)
        else:
            content, is_error, signal = self._git_diff(tool_input)
        invalid = is_error and signal in {"invalid_tool", "bad_tool_args", "mutating_tool", "unsupported_tool"}
        result = self._tool_result_message(block, content, is_error=is_error, signal=signal)
        record = ToolExecutionRecord(
            step_index=step_index,
            tool_name=str(block.get("name", "")).strip(),
            canonical_tool_name=canonical_name,
            tool_input=dict(tool_input),
            invalid=invalid,
            is_error=is_error,
            content_preview=_truncate(content),
        )
        return result, record.invalid, record

    def modified_files(self) -> tuple[str, ...]:
        completed = _git(["status", "--porcelain", "--untracked-files=no"], cwd=self.workspace_root)
        if completed.returncode != 0:
            return ()
        files: list[str] = []
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            files.append(line[3:].strip())
        return tuple(sorted(files))

    def _canonical_tool_name(self, name: str) -> str:
        normalized = name.lower().strip()
        if normalized in {"view_file", "read", "read_file"}:
            return "view_file"
        if normalized in {"grep_search", "search", "grep", "rg"}:
            return "grep_search"
        if normalized in {"list_dir", "ls"}:
            return "list_dir"
        if normalized in {"edit_file", "edit", "replace_file_content"}:
            return "edit_file"
        if normalized in {"run_tests", "run", "run_terminal", "bash"}:
            return "run_tests"
        if normalized in {"git_diff", "diff"}:
            return "git_diff"
        return ""

    def _tool_result_message(
        self,
        block: dict[str, Any],
        content: str,
        *,
        is_error: bool = False,
        signal: str = "",
    ) -> dict[str, Any]:
        result_block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": str(block.get("id", "")).strip(),
            "content": _truncate(content, self.max_output_chars),
        }
        if is_error:
            result_block["is_error"] = True
        if signal:
            result_block["contract_signal"] = signal
        return {
            "role": "user",
            "content": [result_block],
        }

    def _safe_path(self, raw_path: str | None) -> Path:
        path = _resolved_path(self.workspace_root, raw_path)
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise RuntimeError(f"path escapes workspace: {raw_path}") from exc
        return path

    def _edit_file(self, tool_input: dict[str, Any]) -> tuple[str, bool, str]:
        raw_path = (
            tool_input.get("path") or tool_input.get("file_path") or tool_input.get("target") or tool_input.get("text")
        )
        if not raw_path:
            return "ERROR: path is required", True, "bad_tool_args"
        path = self._safe_path(str(raw_path))
        old_string = tool_input.get("old_string") or tool_input.get("search") or tool_input.get("before")
        new_string = tool_input.get("new_string") or tool_input.get("replace") or tool_input.get("after")
        content = tool_input.get("content")
        if old_string is None and new_string is None and content is None:
            return "ERROR: edit_file requires old/new strings or full content", True, "bad_tool_args"
        if path.exists():
            original = path.read_text()
        else:
            original = ""
        if old_string is not None and new_string is not None:
            if str(old_string) not in original:
                return f"ERROR: search string not found in {path.name}", True, "bad_tool_args"
            updated = original.replace(str(old_string), str(new_string), 1)
        elif content is not None:
            updated = str(content)
        else:
            return "ERROR: edit_file requires a replacement payload", True, "bad_tool_args"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated)
        return f"Edited {_safe_relative_path(self.workspace_root, path)}", False, ""

    def _run_tests(self, tool_input: dict[str, Any]) -> tuple[str, bool, str]:
        command = str(tool_input.get("command") or tool_input.get("cmd") or tool_input.get("text") or "").strip()
        if not command:
            return "ERROR: run_tests requires a pytest command", True, "bad_tool_args"
        if not _TEST_COMMAND_RE.match(command):
            return "ERROR: only pytest commands are allowed in run_tests", True, "invalid_tool"
        completed = subprocess.run(
            command,
            cwd=self.workspace_root,
            shell=False,
            executable="/bin/zsh",
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        output = completed.stdout.strip()
        if completed.stderr.strip():
            output = f"{output}\n{completed.stderr.strip()}".strip()
        content = f"$ {command}\n{output}".strip()
        is_error = completed.returncode != 0
        signal = "command_failed" if is_error else ""
        return content, is_error, signal

    def _git_diff(self, tool_input: dict[str, Any]) -> tuple[str, bool, str]:
        paths: list[str] = []
        if tool_input.get("path"):
            paths.append(str(tool_input["path"]))
        if isinstance(tool_input.get("paths"), list):
            paths.extend(str(item) for item in tool_input.get("paths", []))
        args = ["diff", "--unified=3"]
        if paths:
            args.extend(["--", *paths])
        completed = _git(args, cwd=self.workspace_root)
        if completed.returncode not in {0, 1}:
            return f"ERROR: git diff failed: {completed.stderr.strip()}", True, "command_failed"
        return completed.stdout.strip() or "(no diff)", False, ""


_XML_TOOL_PATTERN = re.compile(
    r"<tool_name>\s*(\w+)\s*</tool_name>",
    re.IGNORECASE | re.DOTALL,
)
_XML_PARAM_PATTERN = re.compile(
    r'<parameter\s+name\s*=\s*"([^"]*)">\s*(.*?)\s*</parameter>',
    re.IGNORECASE | re.DOTALL,
)
_TOK_TOOL_PATTERN = re.compile(
    r"@Tool\s+(\w+)\s*\{([^}]*)\}",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_TOOL_USE_PATTERN = re.compile(
    r"Tool\s+use\s*\(\s*(\w+)\s*\)\s*:\s*(\{[^}]*\})",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.IGNORECASE | re.DOTALL,
)


def _parse_key_value_object(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for m in re.finditer(r'(?:(\w+)\s*:\s*)?["\']?([^"\',:}\s]+)["\']?', text):
        key = m.group(1)
        value = m.group(2).strip()
        if key:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def _extract_text_tool_calls(
    raw_response: str,
    allowed_tools: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed_lower = {t.lower() for t in allowed_tools}
    blocks: list[dict[str, Any]] = []
    for tool_match in _XML_TOOL_PATTERN.finditer(raw_response):
        tool_name = tool_match.group(1).strip()
        if tool_name.lower() not in allowed_lower:
            continue
        after = raw_response[tool_match.end() :]
        next_tool = _XML_TOOL_PATTERN.search(after)
        search_region = after[: next_tool.start()] if next_tool else after
        tool_input: dict[str, Any] = {}
        for param_match in _XML_PARAM_PATTERN.finditer(search_region):
            key = param_match.group(1).strip()
            value = param_match.group(2).strip()
            tool_input[key] = value
        blocks.append(
            {
                "type": "tool_use",
                "id": f"text_extracted_{len(blocks)}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    for tok_match in _TOK_TOOL_PATTERN.finditer(raw_response):
        tool_name = tok_match.group(1).strip()
        if tool_name.lower() not in allowed_lower:
            continue
        tool_input = _parse_key_value_object(tok_match.group(2))
        blocks.append(
            {
                "type": "tool_use",
                "id": f"tok_extracted_{len(blocks)}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    for text_match in _TEXT_TOOL_USE_PATTERN.finditer(raw_response):
        tool_name = text_match.group(1).strip()
        if tool_name.lower() not in allowed_lower:
            continue
        tool_input = _parse_key_value_object(text_match.group(2))
        blocks.append(
            {
                "type": "tool_use",
                "id": f"text_use_extracted_{len(blocks)}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    for fenced_match in _FENCED_JSON_BLOCK_PATTERN.finditer(raw_response):
        payload_raw = fenced_match.group(1).strip()
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue

        payload_items: list[dict[str, Any]]
        if isinstance(payload, dict):
            payload_items = [payload]
        elif isinstance(payload, list):
            payload_items = [item for item in payload if isinstance(item, dict)]
        else:
            payload_items = []

        for item in payload_items:
            tool_name = str(
                item.get("tool") or item.get("tool_name") or item.get("name") or item.get("function") or ""
            ).strip()
            if not tool_name or tool_name.lower() not in allowed_lower:
                continue

            raw_input = item.get("input")
            if not isinstance(raw_input, dict):
                raw_input = item.get("args")
            if not isinstance(raw_input, dict):
                raw_input = item.get("parameters")
            if not isinstance(raw_input, dict):
                raw_input = {
                    key: value
                    for key, value in item.items()
                    if key not in {"tool", "tool_name", "name", "function", "input", "args", "parameters"}
                }

            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"fenced_json_extracted_{len(blocks)}",
                    "name": tool_name,
                    "input": dict(raw_input),
                }
            )
    return blocks


class ToolLoopExecutor:
    """Run a live benchmark task with the shared conversation transport."""

    def __init__(self, runner: LiveBenchmarkRunner, *, catalog_root: Path) -> None:
        self.runner = runner
        self.evaluator = FamilyEvaluator(catalog_root=catalog_root)

    def run_task(
        self,
        materialized: MaterializedBenchmarkTask,
        *,
        output_root: Path,
    ) -> BenchmarkTaskRunResult:
        task = materialized.task
        workspace_root = Path(materialized.workspace_root)
        timeout_seconds = int(
            task.success_evaluator.get("hidden_tests_timeout_seconds", task.time_budget_minutes * 60) or 120
        )
        tool_executor = BenchmarkToolExecutor(
            workspace_root,
            allowed_tools=task.allowed_tools,
            timeout_seconds=max(30, timeout_seconds),
        )
        condition = materialized.condition
        bridge_session = (
            BridgeSession(memory_dir=workspace_root / ".tok_benchmark_memory") if condition == "tok-universal" else None
        )
        session = bridge_session.runtime_session if bridge_session is not None else RuntimeSession()
        conversation: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        turns: list[dict[str, Any]] = []
        tool_records: list[ToolExecutionRecord] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0
        invalid_tool_calls = 0
        tool_calls = 0
        reacquisition_events = 0
        answer_text = ""
        raw_response = ""
        clean_exit = False
        notes: list[str] = []

        for step_index in range(1, task.step_budget + 1):
            step = self._run_conversation_step_with_retry(
                conversation=conversation,
                system_prompt=self._system_prompt(task, condition),
                mode=condition,
                session=session,
                bridge_session=bridge_session,
                allowed_tools=task.allowed_tools,
                notes=notes,
                step_index=step_index,
            )
            total_prompt_tokens += int(step.provider_usage.prompt_tokens)
            total_completion_tokens += int(step.provider_usage.completion_tokens)
            total_latency_ms += float(step.provider_usage.latency_ms)
            reacquisition_events += _count_reacquisition_events(step.response_metrics["response_behavior_signals"])
            tool_blocks = [dict(block) for block in step.content_blocks if block.get("type") == "tool_use"]
            extracted_from_text = False

            if not tool_blocks and step.raw_response:
                text_tool_blocks = _extract_text_tool_calls(step.raw_response, task.allowed_tools)
                if text_tool_blocks:
                    tool_blocks.extend(text_tool_blocks)
                    extracted_from_text = True
                    notes.append(f"text_tool_extraction_step_{step_index}")

            if extracted_from_text:
                assistant_content: str | list[dict[str, Any]] = copy.deepcopy(tool_blocks)
            else:
                assistant_content = _assistant_message_content(
                    step.content_blocks, step.visible_response or step.raw_response
                )
            conversation.append({"role": "assistant", "content": assistant_content})

            turns.append(
                {
                    "step": step_index,
                    "provider_usage": asdict(step.provider_usage),
                    "visible_response": step.visible_response,
                    "raw_response": step.raw_response,
                    "response_metrics": dict(step.response_metrics),
                    "compression_metrics": dict(step.compression_metrics),
                    "tool_use_count": len(tool_blocks),
                }
            )
            raw_response = step.raw_response
            if not tool_blocks:
                proposed_answer = step.visible_response or step.raw_response
                if self._requires_more_tooling(task, tool_calls):
                    notes.append(f"premature_final_step_{step_index}")
                    conversation.append(
                        {
                            "role": "user",
                            "content": self._tooling_recovery_prompt(task=task, tool_calls=tool_calls),
                        }
                    )
                    continue
                answer_text = proposed_answer
                clean_exit = True
                break

            for block in tool_blocks:
                tool_message, invalid, record = tool_executor.execute_tool(block, step_index=step_index)
                tool_calls += 1
                invalid_tool_calls += int(invalid)
                tool_records.append(record)
                conversation.append(tool_message)
        else:
            notes.append("step_budget_exhausted")

        evaluation = self.evaluator.evaluate(
            materialized,
            answer_text=answer_text,
            clean_exit=clean_exit,
            invalid_tool_calls=invalid_tool_calls,
            tool_calls=tool_calls,
            tool_records=tool_records,
            workspace_root=workspace_root,
        )
        provider_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "latency_ms": round(total_latency_ms, 2),
        }
        result = BenchmarkTaskRunResult(
            lane_id=materialized.lane.id,
            condition=condition,
            task_id=task.id,
            family=task.family,
            repeat_index=materialized.repeat_index,
            workspace_root=str(workspace_root),
            answer_text=answer_text,
            raw_response=raw_response,
            provider_usage=provider_usage,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            reacquisition_events=reacquisition_events,
            clean_exit=clean_exit,
            modified_files=tool_executor.modified_files(),
            tool_records=tuple(tool_records),
            turns=tuple(turns),
            evaluation=evaluation,
            notes=tuple(notes),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "run.json").write_text(json.dumps(result.to_dict(), indent=2))
        return result

    def _is_transient_connection_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return (
            "connection error" in message
            or "timed out" in message
            or "timeout" in message
            or "temporarily unavailable" in message
            or "rate limit" in message
            or "429" in message
            or "502" in message
            or "503" in message
            or "504" in message
        )

    def _run_conversation_step_with_retry(
        self,
        *,
        conversation: list[dict[str, Any]],
        system_prompt: str,
        mode: str,
        session: RuntimeSession,
        bridge_session: BridgeSession | None,
        allowed_tools: tuple[str, ...],
        notes: list[str],
        step_index: int,
    ) -> Any:
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return self.runner.run_conversation_step(
                    conversation=conversation,
                    system_prompt=system_prompt,
                    mode=mode,
                    session=session,
                    bridge_session=bridge_session,
                    allowed_tools=allowed_tools,
                )
            except Exception as exc:
                if not self._is_transient_connection_error(exc) or attempt >= max_attempts:
                    raise
                notes.append(f"transient_connection_retry_step_{step_index}_attempt_{attempt}")
                time.sleep(0.5 * attempt)
        msg = "retry loop exhausted without raising final exception"
        raise RuntimeError(msg)

    def _requires_more_tooling(self, task: BenchmarkTaskManifest, tool_calls: int) -> bool:
        if task.family == "repo_grounding":
            return False
        if task.family == "execution_patch":
            return tool_calls < 1
        return False

    def _tooling_recovery_prompt(self, *, task: BenchmarkTaskManifest, tool_calls: int) -> str:
        if task.family == "repo_grounding":
            return (
                "Continue investigating with the allowed tools as needed before finalizing. "
                f"You currently have {tool_calls} tool calls. Prefer grounded citations in your final answer when possible."
            )
        if task.family == "execution_patch":
            return (
                "Continue using the allowed tools before finalizing. "
                "Run the necessary read/edit/test steps to verify the fix, then provide the final answer."
            )
        return "Continue using the allowed tools and then provide the final answer."

    def _system_prompt(self, task: BenchmarkTaskManifest, condition: str = "baseline") -> str:
        tool_list = ", ".join(f"`{tool}`" for tool in task.allowed_tools)
        tool_params = (
            "Tool parameters: "
            "view_file(path, start?, end?) — start/end are 1-based line numbers for reading a range. "
            "grep_search(pattern, path?) — search file contents. "
            "list_dir(path). "
        )
        del condition
        tool_instruction = "When you need a tool, call it using the function calling interface. " + tool_params
        shared = (
            "You are running a controlled benchmark task in a local repository workspace. "
            f"Use only these tools: {tool_list}. "
            + tool_instruction
            + "Do not invent file contents, tests, or evidence. "
            "If a tool errors, recover using the allowed tools. "
        )
        if task.family == "execution_patch":
            return (
                shared
                + "Fix the bug, keep changes inside the allowed files, and only finish after you have run the most relevant tests."
            )
        if task.family == "repo_grounding":
            return (
                shared
                + "When you have gathered enough information, stop calling tools and write your final answer. "
                + "Prefer ending with an Evidence section in this format when possible:\n"
                + "Evidence:\n"
                + "- <file_path> line <number>: <what this line does>\n"
                + "- <file_path> line <number>: <what this line does>\n"
                + "Keep the answer concise and grounded in required files and symbols from the codebase."
            )
        if task.family == "real_session":
            return shared + f"Continue only to the declared milestone: {task.next_milestone}"
        return shared


class FamilyEvaluator:
    """Deterministic evaluators for each benchmark family."""

    def __init__(self, *, catalog_root: Path | None = None) -> None:
        base = (
            catalog_root.resolve() if catalog_root is not None else Path(__file__).resolve().parents[3] / "benchmarks"
        )
        self.catalog_root = base.resolve()
        self.evaluator_bundle_root = (self.catalog_root / DEFAULT_EVALUATOR_BUNDLE_DIR).resolve()

    def validate_execution_evaluators(self, tasks: tuple[BenchmarkTaskManifest, ...]) -> None:
        for task in tasks:
            if task.family != "execution_patch":
                continue
            self._load_hidden_evaluator_spec(task)

    def evaluate(
        self,
        materialized: MaterializedBenchmarkTask,
        *,
        answer_text: str,
        clean_exit: bool,
        invalid_tool_calls: int,
        tool_calls: int,
        tool_records: list[ToolExecutionRecord],
        workspace_root: Path,
    ) -> TaskEvaluationResult:
        task = materialized.task
        if task.family == "execution_patch":
            return self._evaluate_execution_patch(
                task,
                workspace_root=workspace_root,
                clean_exit=clean_exit,
            )
        if task.family == "repo_grounding":
            return self._evaluate_repo_grounding(
                task,
                answer_text=answer_text,
                clean_exit=clean_exit,
                invalid_tool_calls=invalid_tool_calls,
                tool_calls=tool_calls,
                workspace_root=workspace_root,
            )
        return self._evaluate_real_session(
            task,
            answer_text=answer_text,
            clean_exit=clean_exit,
            invalid_tool_calls=invalid_tool_calls,
            tool_calls=tool_calls,
            workspace_root=workspace_root,
            tool_records=tool_records,
        )

    def compare_pair(
        self,
        *,
        task: BenchmarkTaskManifest,
        lane_id: str,
        repeat_index: int,
        baseline: BenchmarkTaskRunResult,
        candidate: BenchmarkTaskRunResult,
    ) -> BenchmarkComparisonRun:
        paired_result_stable = not any(note == "step_budget_exhausted" for note in (*baseline.notes, *candidate.notes))
        format_contract_violations = tuple(
            note
            for note in candidate.evaluation.notes
            if note in {"answer_contract_sentence_limit", "evidence_block_count", "invalid_citations"}
        )
        min_grounded_steps = int(task.success_evaluator.get("min_grounded_retrieval_steps", 0) or 0)
        tool_engagement_stats = {
            "baseline_tool_calls": baseline.tool_calls,
            "tok_tool_calls": candidate.tool_calls,
            "grounded_retrieval_target": max(0, min_grounded_steps),
            "tok_missing_grounded_retrieval_target": bool(
                task.family == "repo_grounding" and min_grounded_steps > 0 and candidate.tool_calls < min_grounded_steps
            ),
        }
        return BenchmarkComparisonRun(
            lane_id=lane_id,
            task_id=task.id,
            family=task.family,
            repeat_index=repeat_index,
            public_release=task.public_release,
            baseline_success=baseline.success,
            tok_success=candidate.success,
            quality_gate_passed=candidate.success,
            total_token_delta=int(candidate.provider_usage["total_tokens"])
            - int(baseline.provider_usage["total_tokens"]),
            latency_delta_ms=float(candidate.provider_usage["latency_ms"])
            - float(baseline.provider_usage["latency_ms"]),
            reacquisition_events=candidate.reacquisition_events,
            invalid_tool_calls=candidate.invalid_tool_calls,
            paired_result_stable=paired_result_stable,
            baseline_grounding_success=baseline.grounding_success,
            tok_grounding_success=candidate.grounding_success,
            baseline_tool_calls=baseline.tool_calls,
            tok_tool_calls=candidate.tool_calls,
            format_contract_violations=format_contract_violations,
            tool_engagement_stats=tool_engagement_stats,
            matched_completion_pair=baseline.success and candidate.success,
        )

    def _evaluate_execution_patch(
        self,
        task: BenchmarkTaskManifest,
        *,
        workspace_root: Path,
        clean_exit: bool,
    ) -> TaskEvaluationResult:
        hidden_spec = self._load_hidden_evaluator_spec(task)
        timeout_seconds = int(
            hidden_spec.get("timeout_seconds") or task.success_evaluator.get("hidden_tests_timeout_seconds", 60) or 60
        )
        hidden_command = str(hidden_spec.get("command") or "").strip()
        hidden_tests = tuple(hidden_spec.get("selectors") or hidden_spec.get("hidden_tests") or task.hidden_tests)
        if not hidden_command:
            hidden_command = self._pytest_command(hidden_tests)
        hidden_result = self._run_shell(hidden_command, cwd=workspace_root, timeout_seconds=timeout_seconds)
        modified_files = self._modified_files(workspace_root)
        allowed_paths_ok = all(path in task.allowed_paths for path in modified_files)
        hidden_ok = hidden_result.returncode == 0
        success = hidden_ok and allowed_paths_ok
        notes: list[str] = []
        if not hidden_ok:
            notes.append("hidden_tests_failed")
        if not allowed_paths_ok:
            notes.append("allowed_path_check_failed")
        if task.success_evaluator.get("clean_exit_required") and not clean_exit:
            notes.append("clean_exit_preferred")
        return TaskEvaluationResult(
            success=success,
            grounding_success=success,
            details={
                "hidden_tests_command": hidden_command,
                "hidden_tests_returncode": hidden_result.returncode,
                "hidden_tests_output": _truncate(
                    _normalize_pytest_output((hidden_result.stdout or "") + (hidden_result.stderr or ""))
                ),
                "evaluator_spec": task.evaluator_spec_ref(),
                "modified_files": list(modified_files),
                "allowed_paths_ok": allowed_paths_ok,
                "clean_exit": clean_exit,
            },
            notes=tuple(notes),
        )

    def _evaluate_repo_grounding(
        self,
        task: BenchmarkTaskManifest,
        *,
        answer_text: str,
        clean_exit: bool,
        invalid_tool_calls: int,
        tool_calls: int,
        workspace_root: Path,
    ) -> TaskEvaluationResult:
        cleaned_answer = _clean_answer_text(answer_text)
        payload = self._load_gold_answer_payload(task, workspace_root=workspace_root)
        required_files = tuple(payload.get("required_files") or task.required_files)
        required_symbols = tuple(payload.get("required_symbols") or task.required_symbols)
        supporting_spans = tuple(payload.get("supporting_spans") or [dict(item) for item in task.supporting_spans])
        sentence_limit = 6
        sentence_ok = _sentence_count(cleaned_answer) <= sentence_limit
        evidence = _evidence_lines(cleaned_answer)
        evidence_count_ok = 2 <= len(evidence) <= 4
        citations_valid = self._citations_valid(evidence, supporting_spans)
        file_hit = any(_file_mentioned(path, cleaned_answer) for path in required_files)
        symbol_hit = any(symbol in cleaned_answer for symbol in required_symbols)
        min_steps = int(task.success_evaluator.get("min_grounded_retrieval_steps", 2) or 2)
        grounded_steps_ok = tool_calls >= min_steps
        completion_signals_ok = clean_exit and file_hit and symbol_hit and invalid_tool_calls == 0
        success = completion_signals_ok
        grounding_success = completion_signals_ok
        notes: list[str] = []
        if not sentence_ok:
            notes.append("answer_contract_sentence_limit")
        if not evidence_count_ok:
            notes.append("evidence_block_count")
        if not citations_valid:
            notes.append("invalid_citations")
        if not file_hit:
            notes.append("required_file_missing")
        if not symbol_hit:
            notes.append("required_symbol_missing")
        if not grounded_steps_ok:
            notes.append("grounded_retrieval_steps_missing")
        if invalid_tool_calls:
            notes.append("invalid_tool_calls_present")
        return TaskEvaluationResult(
            success=success,
            grounding_success=grounding_success,
            details={
                "completion_signals_ok": completion_signals_ok,
                "sentence_count": _sentence_count(cleaned_answer),
                "evidence_lines": evidence,
                "citations_valid": citations_valid,
                "required_file_hit": file_hit,
                "required_symbol_hit": symbol_hit,
                "tool_calls": tool_calls,
                "invalid_tool_calls": invalid_tool_calls,
                "clean_exit": clean_exit,
                "format_contract_violations": [
                    note
                    for note, ok in (
                        ("answer_contract_sentence_limit", sentence_ok),
                        ("evidence_block_count", evidence_count_ok),
                        ("invalid_citations", citations_valid),
                    )
                    if not ok
                ],
                "tool_engagement_stats": {
                    "tool_calls": tool_calls,
                    "min_grounded_retrieval_steps": min_steps,
                    "grounded_retrieval_steps_met": grounded_steps_ok,
                },
            },
            notes=tuple(notes),
        )

    def _evaluate_real_session(
        self,
        task: BenchmarkTaskManifest,
        *,
        answer_text: str,
        clean_exit: bool,
        invalid_tool_calls: int,
        tool_calls: int,
        workspace_root: Path,
        tool_records: list[ToolExecutionRecord],
    ) -> TaskEvaluationResult:
        milestone_type = str(task.success_evaluator.get("milestone_type") or task.episode_type or "").strip()
        invalid_limit = int(task.success_evaluator.get("invalid_tool_calls_limit", 0) or 0)
        if milestone_type in {"grounded_answer", "bug_diagnosis"}:
            grounding = self._evaluate_repo_grounding(
                task,
                answer_text=answer_text,
                clean_exit=clean_exit,
                invalid_tool_calls=invalid_tool_calls,
                tool_calls=tool_calls,
                workspace_root=workspace_root,
            )
            success = grounding.success and invalid_tool_calls <= invalid_limit
            details = dict(grounding.details)
            details["milestone_type"] = milestone_type
            notes = list(grounding.notes)
            if invalid_tool_calls > invalid_limit:
                notes.append("invalid_tool_call_limit_exceeded")
            return TaskEvaluationResult(
                success=success, grounding_success=grounding.grounding_success, details=details, notes=tuple(notes)
            )

        execution = self._evaluate_execution_patch(task, workspace_root=workspace_root, clean_exit=clean_exit)
        success = execution.success and invalid_tool_calls <= invalid_limit
        details = dict(execution.details)
        details["milestone_type"] = milestone_type or "patch_completion"
        details["tool_records"] = [record.to_dict() for record in tool_records]
        notes = list(execution.notes)
        if invalid_tool_calls > invalid_limit:
            notes.append("invalid_tool_call_limit_exceeded")
        return TaskEvaluationResult(success=success, grounding_success=success, details=details, notes=tuple(notes))

    def _pytest_command(self, tests: tuple[str, ...]) -> str:
        if not tests:
            msg = "execution_patch evaluator requires selectors or an explicit command"
            raise RuntimeError(msg)
        return "python -m pytest -q " + " ".join(tests)

    def _load_hidden_evaluator_spec(self, task: BenchmarkTaskManifest) -> dict[str, Any]:
        evaluator_spec = task.evaluator_spec_ref()
        if evaluator_spec:
            raw = Path(evaluator_spec)
            candidate = raw if raw.is_absolute() else (self.catalog_root / raw)
            candidate = candidate.resolve()
            if not candidate.is_file():
                msg = f"task {task.id} references evaluator spec '{evaluator_spec}', but no file exists at {candidate}"
                raise RuntimeError(msg)
            payload: dict[str, Any] = json.loads(candidate.read_text())
            payload["source"] = str(candidate)
            return payload
        if task.hidden_tests:
            return {"selectors": list(task.hidden_tests)}
        return {}

    def _load_gold_answer_payload(
        self,
        task: BenchmarkTaskManifest,
        *,
        workspace_root: Path | None,
    ) -> dict[str, Any]:
        gold_answer_path = str(task.family_payload.get("gold_answer_path") or "").strip()
        if not gold_answer_path:
            return dict(task.effective_family_payload())
        candidate_bases: list[Path] = []
        if workspace_root is not None:
            candidate_bases.append(workspace_root)
            candidate_bases.append(workspace_root.parent)
        candidate_bases.extend(
            [
                Path(task.asset_dir),
                Path.cwd() / task.asset_dir,
                Path(__file__).resolve().parents[3] / "benchmarks" / task.asset_dir,
            ]
        )
        for base in candidate_bases:
            candidate = base / gold_answer_path
            if candidate.exists():
                result = json.loads(candidate.read_text())
                if isinstance(result, dict):
                    return result
                msg = f"gold answer at {candidate} is not a dict"
                raise RuntimeError(msg)
        return dict(task.effective_family_payload())

    def _run_shell(self, command: str, *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            shell=False,
            executable="/bin/zsh",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

    def _modified_files(self, workspace_root: Path) -> tuple[str, ...]:
        completed = _git(["status", "--porcelain", "--untracked-files=no"], cwd=workspace_root)
        if completed.returncode != 0:
            return ()
        paths = [line[3:].strip() for line in completed.stdout.splitlines() if line.strip()]
        return tuple(sorted(paths))

    def _citations_valid(self, citations: list[str], supporting_spans: tuple[dict[str, Any], ...]) -> bool:
        if not citations:
            return False
        line_ref_pattern = re.compile(r"(?:\bline\s+\d+\b|:\d+\b|#L\d+\b)", re.IGNORECASE)
        for citation in citations:
            if not any(
                (str(span.get("file", "")) in citation or Path(str(span.get("file", ""))).name in citation)
                and (
                    str(span.get("anchor", "")) in citation
                    or any(
                        token and token in citation
                        for token in re.findall(
                            r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                            str(span.get("anchor", "")),
                        )
                    )
                    or bool(line_ref_pattern.search(citation))
                )
                for span in supporting_spans
            ):
                return False
        return True


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
        local_failure="task_execution_failed",
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

    output_root.mkdir(parents=True, exist_ok=True)
    evaluator.validate_execution_evaluators(tasks)
    for task in tasks:
        for repeat_index in range(1, max(1, repeats) + 1):
            pair_results: dict[str, BenchmarkTaskRunResult] = {}
            for condition in ("baseline", "tok-universal"):
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
                candidate=pair_results["tok-universal"],
            )
            comparison_runs.append(comparison)
            compare_path = output_root / "tasks" / task.id / f"repeat_{repeat_index}" / "compare.json"
            compare_path.write_text(json.dumps(comparison.to_dict(), indent=2))

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
